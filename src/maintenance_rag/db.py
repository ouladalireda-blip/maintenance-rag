"""Accès à la base Supabase/Postgres (psycopg) : écriture des documents et passages, appel de la
fonction SQL hybrid_search.

Écriture d'un document en UNE transaction : upsert du document par ``slug``, suppression de
ses anciens passages, insertion des nouveaux. Si une étape échoue, la transaction est annulée
en entier (``connection.transaction()`` fait le rollback) : la base garde l'état précédent.

Toutes les requêtes sont paramétrées (``%s``) : aucune valeur n'est jamais insérée dans le
texte SQL. Les vecteurs sont transmis sous forme textuelle « [x1,x2,…] » et convertis par
``::extensions.vector`` côté base (pas de dépendance au paquet pgvector).
"""

from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, LiteralString, Protocol, Self


class Cursor(Protocol):
    """Sous-ensemble de psycopg.Cursor utilisé ici."""

    def __enter__(self) -> Self: ...

    def __exit__(self, *exc: object) -> None: ...

    def execute(self, query: LiteralString, params: Sequence[Any] | None = None) -> Any: ...

    def executemany(self, query: LiteralString, params_seq: Iterable[Sequence[Any]]) -> None: ...

    def fetchone(self) -> Any: ...


class Connection(Protocol):
    """Sous-ensemble de psycopg.Connection utilisé ici."""

    def cursor(self) -> AbstractContextManager[Any]: ...  # entré, il fournit un Cursor

    def transaction(self) -> AbstractContextManager[Any]: ...


@dataclass(frozen=True)
class DocumentRow:
    """Ligne de public.documents (colonnes de 0001_init.sql)."""

    slug: str
    title: str
    source_url: str | None
    license: str
    language: str
    version: str | None
    machine: str | None
    page_offset: int
    sha256: str


@dataclass(frozen=True)
class ChunkRow:
    """Ligne de public.chunks (0001_init.sql + has_warning de 0002)."""

    chunk_index: int
    section_title: str
    page_start: int
    page_end: int
    content: str
    token_count: int
    identifiers: tuple[str, ...]
    embedding: tuple[float, ...]
    has_warning: bool


UPSERT_DOCUMENT: LiteralString = """
    insert into public.documents
        (slug, title, source_url, license, language, version, machine, page_offset, sha256)
    values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    on conflict (slug) do update set
        title       = excluded.title,
        source_url  = excluded.source_url,
        license     = excluded.license,
        language    = excluded.language,
        version     = excluded.version,
        machine     = excluded.machine,
        page_offset = excluded.page_offset,
        sha256      = excluded.sha256,
        ingested_at = now()
    returning id
"""

DELETE_CHUNKS: LiteralString = "delete from public.chunks where document_id = %s"

INSERT_CHUNK: LiteralString = """
    insert into public.chunks
        (document_id, chunk_index, section_title, page_start, page_end, content, token_count,
         identifiers, embedding, has_warning)
    values (%s, %s, %s, %s, %s, %s, %s, %s, %s::extensions.vector, %s)
"""


def vector_literal(vector: Sequence[float]) -> str:
    """Représentation textuelle pgvector : « [x1,x2,…] » (repr exact des flottants)."""
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def document_params(document: DocumentRow) -> tuple[Any, ...]:
    return (
        document.slug,
        document.title,
        document.source_url,
        document.license,
        document.language,
        document.version,
        document.machine,
        document.page_offset,
        document.sha256,
    )


def chunk_params(document_id: Any, chunk: ChunkRow) -> tuple[Any, ...]:
    return (
        document_id,
        chunk.chunk_index,
        chunk.section_title,
        chunk.page_start,
        chunk.page_end,
        chunk.content,
        chunk.token_count,
        list(chunk.identifiers),
        vector_literal(chunk.embedding),
        chunk.has_warning,
    )


def write_document(
    connection: Connection, document: DocumentRow, chunks: Sequence[ChunkRow]
) -> Any:
    """Remplace le document et tous ses passages en une transaction ; renvoie son id."""
    with connection.transaction(), connection.cursor() as opened:
        cursor: Cursor = opened
        cursor.execute(UPSERT_DOCUMENT, document_params(document))
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"upsert du document {document.slug} : aucun id renvoyé")
        document_id = row[0]
        cursor.execute(DELETE_CHUNKS, (document_id,))
        cursor.executemany(INSERT_CHUNK, [chunk_params(document_id, chunk) for chunk in chunks])
    return document_id
