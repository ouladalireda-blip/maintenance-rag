"""Métriques d'évaluation : rappel de récupération (gold), exactitude des citations, taux de refus
correct / incorrect par catégorie.

Fonctions pures, sans base ni réseau (tests : tests/test_metrics.py).

Un passage est « le bon » pour une question (``hit``) si la citation de référence y figure mot
pour mot (``quote_matches``, mêmes règles que le garde-fou 2) ET si la page imprimée attendue
est dans sa plage de pages imprimées. Le rang d'une question est la position (dès 1) du
premier bon passage dans les résultats, ``None`` s'il n'est pas dans le top 10.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from maintenance_rag.normalize import quote_matches

TOP_K = 10


class Result(Protocol):
    @property
    def content(self) -> str: ...

    @property
    def printed_page_start(self) -> int | None: ...

    @property
    def printed_page_end(self) -> int | None: ...


def hit(result: Result, question: Mapping[str, Any], min_chars: int = 20) -> bool:
    """Vrai si le passage contient la citation ET couvre la page imprimée attendue."""
    start, end = result.printed_page_start, result.printed_page_end
    page = question["expected_page"]
    if start is None or end is None or not start <= page <= end:
        return False
    return quote_matches(question["exact_quote"], result.content, min_chars)


def first_hit_rank(
    results: Sequence[Result], question: Mapping[str, Any], min_chars: int = 20, k: int = TOP_K
) -> int | None:
    """Rang (dès 1) du premier bon passage parmi les ``k`` premiers, ``None`` sinon."""
    for rank, result in enumerate(results[:k], start=1):
        if hit(result, question, min_chars):
            return rank
    return None


def _check(ranks: Sequence[int | None]) -> None:
    if not ranks:
        raise ValueError("aucune question à évaluer")
    if any(rank is not None and rank < 1 for rank in ranks):
        raise ValueError("un rang commence à 1")


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Part des questions dont le bon passage est dans les ``k`` premiers."""
    if k < 1:
        raise ValueError("k doit valoir au moins 1")
    _check(ranks)
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mrr(ranks: Sequence[int | None]) -> float:
    """Moyenne des inverses de rang (0 si le bon passage est absent du top 10)."""
    _check(ranks)
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)
