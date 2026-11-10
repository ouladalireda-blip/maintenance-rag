"""Tests des métriques d'évaluation sur des cas faits main (sans base, sans réseau)."""

from dataclasses import dataclass

import pytest
from evals.metrics import (
    Candidate,
    Paired,
    e2e_criteria,
    first_hit_rank,
    hit,
    mrr,
    paired_comparison,
    percentile,
    recall_at_k,
    select_config,
    separation,
    spread,
)

QUESTION = {
    "id": "Q1",
    "expected_page": 14,
    "exact_quote": "la mise à la terre est réalisée après la vérification",
}


@dataclass(frozen=True)
class FakeResult:
    content: str
    printed_page_start: int | None
    printed_page_end: int | None


GOOD = FakeResult("Texte. La mise à la terre est réalisée après la vérification.", 14, 14)


def test_hit_requires_quote_and_page() -> None:
    assert hit(GOOD, QUESTION)
    assert hit(FakeResult(GOOD.content, 13, 15), QUESTION)  # page dans la plage
    assert not hit(FakeResult(GOOD.content, 15, 16), QUESTION)  # bonne citation, mauvaise page
    assert not hit(FakeResult(GOOD.content, None, None), QUESTION)  # page sans numéro
    assert not hit(FakeResult("La mise à la terre est réalisée avant.", 14, 14), QUESTION)


def test_hit_uses_quote_normalization() -> None:
    spaced = FakeResult("LA MISE À LA TERRE est réa\xadlisée après la vérifi-\ncation", 14, 14)
    assert hit(spaced, QUESTION)


def test_first_hit_rank() -> None:
    other = FakeResult("Autre passage sans la citation attendue.", 14, 14)
    assert first_hit_rank([other, other, GOOD], QUESTION) == 3
    assert first_hit_rank([other] * 10 + [GOOD], QUESTION) is None  # hors top 10
    assert first_hit_rank([], QUESTION) is None


def test_recall_at_k() -> None:
    ranks = [1, 3, None, 10]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.5
    assert recall_at_k(ranks, 10) == 0.75


def test_mrr() -> None:
    assert mrr([1, 2, None, 4]) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert mrr([None, None]) == 0.0
    assert mrr([1]) == 1.0


@pytest.mark.parametrize("bad", [[], [0], [-1, 2]])
def test_invalid_ranks(bad: list[int | None]) -> None:
    with pytest.raises(ValueError):
        mrr(bad)
    with pytest.raises(ValueError):
        recall_at_k(bad, 1)


def test_invalid_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k([1], 0)


def test_paired_comparison() -> None:
    a = [1, 3, 2, None, None, 5]
    b = [2, 3, 1, 4, None, None]
    # q1 victoire, q2 égalité, q3 défaite, q4 défaite (absent = pire), q5 égalité, q6 victoire
    assert paired_comparison(a, b) == Paired(wins=2, losses=2, ties=2)
    assert paired_comparison(b, a) == Paired(wins=2, losses=2, ties=2)


def test_paired_comparison_absent_is_worst() -> None:
    assert paired_comparison([None], [20]) == Paired(0, 1, 0)  # même un rang 20 bat l'absence
    assert paired_comparison([20], [None]) == Paired(1, 0, 0)
    assert paired_comparison([None, None], [None, None]) == Paired(0, 0, 2)


@pytest.mark.parametrize(("a", "b"), [([1], [1, 2]), ([], []), ([0], [1]), ([1], [-1])])
def test_paired_comparison_invalid(a: list[int | None], b: list[int | None]) -> None:
    with pytest.raises(ValueError):
        paired_comparison(a, b)


def _candidate(name: str, recall: float, f: float, match_count: int, mrr_: float) -> Candidate:
    return Candidate(name, recall, f, match_count, mrr_)


def test_select_config_recall_first() -> None:
    best = select_config([_candidate("a", 0.90, 0.0, 10, 0.9), _candidate("b", 0.97, 1.0, 20, 0.5)])
    assert best.name == "b"  # le rappel prime sur f, match_count et MRR


def test_select_config_ties() -> None:
    tied = [
        _candidate("f0.5", 0.97, 0.5, 10, 0.9),
        _candidate("f0-mc20", 0.97, 0.0, 20, 0.9),
        _candidate("f0-mc10", 0.97, 0.0, 10, 0.1),
    ]
    assert select_config(tied).name == "f0-mc10"  # f le plus petit, puis match_count le plus petit
    same = [_candidate("x", 0.97, 0.0, 10, 0.5), _candidate("y", 0.97, 0.0, 10, 0.6)]
    assert select_config(same).name == "y"  # le MRR ne départage qu'un ex æquo restant
    with pytest.raises(ValueError):
        select_config([])


def test_percentile() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == 2.5  # médiane d'un nombre pair de valeurs
    assert percentile(list(map(float, range(1, 21))), 95) == pytest.approx(19.05)
    assert percentile([7.0], 95) == 7.0


@pytest.mark.parametrize(("values", "q"), [([], 50), ([1.0], -1), ([1.0], 101)])
def test_percentile_invalid(values: list[float], q: float) -> None:
    with pytest.raises(ValueError):
        percentile(values, q)


def test_spread() -> None:
    assert spread([0.9, 0.1, 0.5]) == {"min": 0.1, "median": 0.5, "max": 0.9}


def test_separation() -> None:
    values = {"v1": 0.80, "v2": 0.60, "v3": 0.70, "n1": 0.65, "n2": 0.50, "o1": 0.60}
    groups = {"valid": ["v1", "v2", "v3"], "near_miss": ["n1", "n2"], "out_of_scope": ["o1"]}
    result = separation(values, groups)
    assert result["by_group"]["valid"] == {"count": 3, "min": 0.60, "median": 0.70, "max": 0.80}
    assert result["by_group"]["near_miss"]["median"] == pytest.approx(0.575)
    assert result["lowest_reference"] == 0.60
    # strictement au-dessus : o1 (= 0.60) n'est pas compté
    assert result["above_lowest_reference"] == {"near_miss": ["n1"], "out_of_scope": []}


def test_separation_needs_reference_group() -> None:
    with pytest.raises(ValueError):
        separation({"n1": 0.5}, {"near_miss": ["n1"]})
    with pytest.raises(ValueError):
        separation({}, {"valid": []})


def _ok(rows: list[dict[str, object]]) -> list[object]:
    return [row["ok"] for row in rows]


def test_e2e_criteria_thresholds() -> None:
    # Juste aux seuils : 1 réponse dangereuse, 5/29 refusées (17 %), 20/24 bonnes sources (83 %)
    assert _ok(e2e_criteria(1, 5, 29, 20, 24)) == [True, True, True]
    # Juste au-dessus : 2 dangereuses, 6/29 (20,7 %), 19/24 (79 %)
    assert _ok(e2e_criteria(2, 6, 29, 19, 24)) == [False, False, False]
    assert e2e_criteria(0, 6, 29, 19, 24)[1]["value"] == pytest.approx(6 / 29)


def test_e2e_criteria_without_answered_valid() -> None:
    rows = e2e_criteria(0, 29, 29, 0, 0)
    assert rows[2]["value"] is None
    assert rows[2]["ok"] is False


@pytest.mark.parametrize("args", [(0, 0, 0, 0, 0), (0, 30, 29, 0, 0), (0, 0, 29, 5, 4)])
def test_e2e_criteria_invalid(args: tuple[int, int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        e2e_criteria(*args)
