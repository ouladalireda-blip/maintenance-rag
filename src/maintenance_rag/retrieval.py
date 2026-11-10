"""Recherche hybride : préparation de la requête (texte, vecteur, identifiants) et appel de
hybrid_search avec les poids RRF de la configuration.

- ``search`` : une question → les passages classés par hybrid_search (fonction SQL de
  0002_chunks_warning.sql). Les identifiants de la question sont extraits sous forme canonique
  (``extract_identifiers``), comme à l'indexation. Le ``content`` des passages est récupéré
  pour la vérification des citations, mais n'est jamais affiché (``repr=False``).
- ``CachedEmbedder`` : cache disque des vecteurs de QUESTIONS (pas des passages), pour qu'une
  évaluation relancée ne recoûte aucun appel d'embedding.
"""

import hashlib
import json
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, LiteralString, Protocol

from maintenance_rag.db import vector_literal
from maintenance_rag.embeddings import Embedder
from maintenance_rag.identifiers import extract_identifiers

Weights = tuple[float, float, float]  # (semantic, full_text, identifier)

HYBRID_SEARCH: LiteralString = """
    select chunk_id, document_slug, chunk_index, page_start, page_end,
           printed_page_start, printed_page_end, content,
           semantic_rank, full_text_rank, identifier_rank, rrf_score, has_warning
    from public.hybrid_search(
        query_text        => %s,
        query_embedding   => %s::extensions.vector,
        query_identifiers => %s::text[],
        match_count       => %s,
        rrf_k             => %s,
        semantic_weight   => %s,
        full_text_weight  => %s,
        identifier_weight => %s
    )
"""


class Connection(Protocol):
    """Sous-ensemble de psycopg.Connection utilisé ici."""

    def cursor(self) -> AbstractContextManager[Any]: ...


@dataclass(frozen=True)
class SearchResult:
    """Une ligne de hybrid_search. ``content`` n'apparaît pas dans la représentation."""

    chunk_id: int
    document_slug: str
    chunk_index: int
    page_start: int
    page_end: int
    printed_page_start: int | None
    printed_page_end: int | None
    semantic_rank: int | None
    full_text_rank: int | None
    identifier_rank: int | None
    rrf_score: float
    has_warning: bool
    content: str = field(repr=False)


def search(
    question: str,
    embedder: Embedder,
    conn: Connection,
    weights: Weights = (1.0, 1.0, 2.0),
    match_count: int = 10,
    *,
    rrf_k: int = 50,
) -> list[SearchResult]:
    """Passages classés par hybrid_search pour la question, du meilleur au moins bon."""
    vector = embedder.embed([question])[0]
    identifiers = extract_identifiers(question)
    semantic, full_text, identifier = (float(w) for w in weights)
    params = (
        question,
        vector_literal(vector),
        identifiers,
        match_count,
        rrf_k,
        semantic,
        full_text,
        identifier,
    )
    with conn.cursor() as cursor:
        cursor.execute(HYBRID_SEARCH, params)
        rows = cursor.fetchall()
    return [
        SearchResult(
            chunk_id=row[0],
            document_slug=row[1],
            chunk_index=row[2],
            page_start=row[3],
            page_end=row[4],
            printed_page_start=row[5],
            printed_page_end=row[6],
            content=row[7],
            semantic_rank=row[8],
            full_text_rank=row[9],
            identifier_rank=row[10],
            rrf_score=float(row[11]),
            has_warning=bool(row[12]),
        )
        for row in rows
    ]


class CachedEmbedder:
    """Embedder avec cache JSON des vecteurs de questions.

    Clé : sha256 de « modèle + "\\n" + texte ». Seuls les textes absents du cache sont
    envoyés à l'embedder sous-jacent ; le fichier est réécrit après chaque ajout.
    """

    def __init__(self, inner: Embedder, model: str, path: Path) -> None:
        self._inner = inner
        self._model = model
        self._path = path
        self._cache: dict[str, list[float]] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
        self.hits = 0
        self.misses = 0

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self._model}\n{text}".encode()).hexdigest()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [text for text in dict.fromkeys(texts) if self.key(text) not in self._cache]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            for text, vector in zip(missing, self._inner.embed(missing), strict=True):
                self._cache[self.key(text)] = vector
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._cache), encoding="utf-8")
        vectors = [self._cache[self.key(text)] for text in texts]
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError("vecteur en cache de dimension inattendue")
        return vectors
