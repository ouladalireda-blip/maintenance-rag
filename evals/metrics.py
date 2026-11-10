"""Métriques d'évaluation : rappel de récupération (gold), exactitude des citations, taux de refus
correct / incorrect par catégorie.

Fonctions pures, sans base ni réseau (tests : tests/test_metrics.py).

Un passage est « le bon » pour une question (``hit``) si la citation de référence y figure mot
pour mot (``quote_matches``, mêmes règles que le garde-fou 2) ET si la page imprimée attendue
est dans sa plage de pages imprimées. Le rang d'une question est la position (dès 1) du
premier bon passage dans les résultats, ``None`` s'il n'est pas dans le top 10.
"""

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol

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


class Paired(NamedTuple):
    """Comparaison appariée question par question, du point de vue de ``a``."""

    wins: int  # a classe le bon passage strictement mieux que b
    losses: int  # a le classe strictement moins bien
    ties: int  # même rang, ou absent des deux côtés


def paired_comparison(ranks_a: Sequence[int | None], ranks_b: Sequence[int | None]) -> Paired:
    """Victoires / défaites / égalités de ``a`` contre ``b`` ; rang absent = le pire."""
    if len(ranks_a) != len(ranks_b):
        raise ValueError("les deux séries de rangs doivent porter sur les mêmes questions")
    _check(ranks_a)
    _check(ranks_b)
    worst = float("inf")
    wins = losses = ties = 0
    for a, b in zip(ranks_a, ranks_b, strict=True):
        left = worst if a is None else a
        right = worst if b is None else b
        if left < right:
            wins += 1
        elif left > right:
            losses += 1
        else:
            ties += 1
    return Paired(wins, losses, ties)


class Candidate(NamedTuple):
    """Une configuration de la grille, réduite aux critères de la règle de choix."""

    name: str
    recall: float  # rappel au match_count le plus haut de la grille
    full_text_weight: float
    match_count: int
    mrr: float


def select_config(candidates: Sequence[Candidate]) -> Candidate:
    """Règle fixée à l'avance : a) rappel le plus haut ; b) à égalité, poids full-text le plus
    petit puis match_count le plus petit ; c) le MRR ne départage qu'un ex æquo restant."""
    if not candidates:
        raise ValueError("aucune configuration")
    return min(candidates, key=lambda c: (-c.recall, c.full_text_weight, c.match_count, -c.mrr))
