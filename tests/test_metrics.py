"""Tests des métriques d'évaluation sur des cas faits main (sans base, sans réseau)."""

from dataclasses import dataclass

import pytest
from evals.metrics import first_hit_rank, hit, mrr, recall_at_k

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
