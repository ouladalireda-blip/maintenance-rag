"""Tests du re-classement avec FakeReranker (sans modèle, sans base, sans réseau)."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from maintenance_rag.rerank import (
    FakeReranker,
    FlashRankReranker,
    Reranked,
    order_key,
    passage_text,
    rerank,
)
from maintenance_rag.retrieval import SearchResult

BASE = SearchResult(
    chunk_id=0,
    document_slug="inrs-ed6109",
    chunk_index=0,
    section_title="",
    page_start=10,
    page_end=10,
    printed_page_start=8,
    printed_page_end=8,
    semantic_rank=None,
    full_text_rank=None,
    identifier_rank=None,
    rrf_score=0.0,
    has_warning=False,
    content="",
)


def _candidate(chunk_id: int, title: str, content: str, **fields: object) -> SearchResult:
    return replace(BASE, chunk_id=chunk_id, section_title=title, content=content, **fields)  # type: ignore[arg-type]


QUESTION = "consignation électrique du disjoncteur"


def test_sorted_by_decreasing_score() -> None:
    candidates = [
        _candidate(1, "Généralités", "rien à voir"),
        _candidate(2, "5.1. Consignation électrique", "ouvrir le disjoncteur"),
        _candidate(3, "Électricité", "consignation"),
    ]
    reranked = rerank(QUESTION, candidates, FakeReranker())
    assert [r.result.chunk_id for r in reranked] == [2, 3, 1]
    assert [r.score for r in reranked] == sorted((r.score for r in reranked), reverse=True)
    assert reranked[0].score == 0.75  # 3 mots sur 4 (« du » absent)
    assert [r.search_rank for r in reranked] == [2, 3, 1]


def test_ties_broken_by_search_rank_then_chunk_id() -> None:
    # Même texte → même score : l'ordre de la recherche est conservé, quel que soit chunk_id.
    candidates = [_candidate(9, "", "consignation"), _candidate(4, "", "consignation")]
    reranked = rerank(QUESTION, candidates, FakeReranker())
    assert [r.result.chunk_id for r in reranked] == [9, 4]
    assert [r.search_rank for r in reranked] == [1, 2]


class ConstantReranker:
    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        return [0.5] * len(texts)


def test_all_tied_keeps_search_order_deterministically() -> None:
    candidates = [_candidate(i, "", "x") for i in (5, 2, 8)]
    first = rerank(QUESTION, candidates, ConstantReranker())
    assert first == rerank(QUESTION, candidates, ConstantReranker())
    assert [r.result.chunk_id for r in first] == [5, 2, 8]


def test_order_key_falls_back_to_chunk_id() -> None:
    tied = [
        Reranked(result=_candidate(7, "", "a"), score=0.5, search_rank=1),
        Reranked(result=_candidate(3, "", "b"), score=0.5, search_rank=1),
        Reranked(result=_candidate(1, "", "c"), score=0.5, search_rank=2),
        Reranked(result=_candidate(9, "", "d"), score=0.9, search_rank=4),
    ]
    assert [r.result.chunk_id for r in sorted(tied, key=order_key)] == [9, 3, 7, 1]


def test_all_search_fields_are_kept() -> None:
    candidate = _candidate(
        42,
        "5.1. Consignation électrique",
        "ouvrir le disjoncteur",
        document_slug="doc",
        chunk_index=17,
        page_start=16,
        page_end=17,
        printed_page_start=14,
        printed_page_end=15,
        semantic_rank=3,
        full_text_rank=None,
        identifier_rank=1,
        rrf_score=0.06,
        has_warning=True,
    )
    (only,) = rerank(QUESTION, [candidate], FakeReranker())
    assert only.result is candidate  # le SearchResult entier, champ pour champ
    assert only.search_rank == 1
    assert "ouvrir" not in repr(only)  # le texte du passage n'apparaît jamais dans repr


def test_text_given_to_model_is_title_newline_content() -> None:
    fake = FakeReranker()
    rerank(QUESTION, [_candidate(1, "Titre", "Contenu"), _candidate(2, "", "Seul")], fake)
    assert fake.calls == [(QUESTION, ["Titre\nContenu", "Seul"])]
    assert passage_text(_candidate(1, "Titre", "Contenu")) == "Titre\nContenu"


def test_empty_candidates() -> None:
    fake = FakeReranker()
    assert rerank(QUESTION, [], fake) == []
    assert fake.calls == []  # le modèle n'est pas appelé


def test_fake_reranker_is_deterministic() -> None:
    texts = ["consignation", "disjoncteur électrique", "autre"]
    assert FakeReranker().score(QUESTION, texts) == FakeReranker().score(QUESTION, texts)
    assert FakeReranker().score("", texts) == [0.0, 0.0, 0.0]


def test_wrong_number_of_scores_is_refused() -> None:
    class Broken:
        def score(self, question: str, texts: Sequence[str]) -> list[float]:
            return [1.0]

    with pytest.raises(ValueError):
        rerank(QUESTION, [_candidate(1, "", "a"), _candidate(2, "", "b")], Broken())


def test_flashrank_never_downloads(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FlashRankReranker("ms-marco-MultiBERT-L-12", tmp_path, 512)
