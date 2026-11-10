"""Tests de l'ingestion : FakeEmbedder et fausse connexion, aucun accès réseau ni base."""

import copy
import hashlib
import math
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from maintenance_rag import db
from maintenance_rag.chunking import chunk_blocks
from maintenance_rag.config import ManifestEntry
from maintenance_rag.embeddings import FakeEmbedder, OpenAIEmbedder
from maintenance_rag.identifiers import extract_identifiers
from maintenance_rag.ingestion import (
    IngestionError,
    Prepared,
    build_rows,
    ingest,
    verify_sha256,
)
from maintenance_rag.pdf_reader import Block

DIMS = 1536
SHA = "a" * 64


def _entry(sha256: str | None = SHA, path: str = "corpus/test.pdf") -> ManifestEntry:
    return ManifestEntry(
        id="test-doc",
        title="Document de test",
        path=path,
        language="fr",
        version="v1",
        license="Fictif (tests)",
        page_offset=2,
        sha256=sha256,
        layout_profile="inrs",
    )


def _prepared() -> Prepared:
    path = ["1 Chapitre", "1.1. Section"]
    blocks = [
        Block(
            pdf_page=3, printed_page=1, kind="section", text=path[1], bold=True, section_path=path
        ),
        Block(
            pdf_page=3,
            printed_page=1,
            kind="paragraph",
            text="Appliquer la norme NF C18-510 et l'article R. 4411-6.",
            bold=False,
            section_path=path,
        ),
        Block(
            pdf_page=4,
            printed_page=2,
            kind="warning",
            text="Attention !\nNe jamais travailler sous tension.",
            bold=True,
            section_path=[*path[:1], "1.2. Autre section"],
        ),
    ]
    chunks = chunk_blocks(blocks, 350, document_slug="test-doc")
    identifiers = [tuple(extract_identifiers(chunk.content)) for chunk in chunks]
    return Prepared(entry=_entry(), sha256=SHA, chunks=chunks, identifiers=identifiers)


# --------------------------------------------------------------------------------------------
# Fausse base : interprète les trois requêtes de db.py, transaction avec rollback
# --------------------------------------------------------------------------------------------
class FakeDatabase:
    def __init__(self, fail_on_insert: bool = False) -> None:
        self.documents: dict[str, dict[str, Any]] = {}  # slug -> {"id":…, "params":…}
        self.chunks: dict[str, list[tuple[Any, ...]]] = {}  # document_id -> lignes
        self.fail_on_insert = fail_on_insert
        self.queries: list[str] = []

    @contextmanager
    def transaction(self) -> Iterator[None]:
        snapshot = copy.deepcopy((self.documents, self.chunks))
        try:
            yield
        except Exception:
            self.documents, self.chunks = snapshot  # rollback complet
            raise

    def cursor(self) -> "FakeCursor":
        return FakeCursor(self)


class FakeCursor:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database
        self._row: tuple[Any, ...] | None = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, query: str, params: Sequence[Any] | None = None) -> None:
        assert params is not None and query.count("%s") == len(params)
        self.database.queries.append(query)
        if query is db.UPSERT_DOCUMENT:
            slug = params[0]
            known = self.database.documents.get(slug)
            document_id = known["id"] if known else f"uuid-{len(self.database.documents) + 1}"
            self.database.documents[slug] = {"id": document_id, "params": tuple(params)}
            self._row = (document_id,)
        elif query is db.DELETE_CHUNKS:
            self.database.chunks.pop(params[0], None)
        else:
            raise AssertionError("requête inattendue")

    def executemany(self, query: str, params_seq: Sequence[Sequence[Any]]) -> None:
        assert query is db.INSERT_CHUNK
        for params in params_seq:
            assert query.count("%s") == len(params)
            if self.database.fail_on_insert:
                raise RuntimeError("échec simulé de l'insertion")
            self.database.chunks.setdefault(params[0], []).append(tuple(params))

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row


# --------------------------------------------------------------------------------------------
# Lignes et embeddings
# --------------------------------------------------------------------------------------------
def test_rows_are_built_from_chunks() -> None:
    prepared = _prepared()
    document, rows = build_rows(prepared, FakeEmbedder(DIMS), DIMS)
    assert document.slug == "test-doc" and document.sha256 == SHA and document.page_offset == 2
    assert [row.chunk_index for row in rows] == [0, 1]
    assert rows[0].identifiers == ("NFC18-510", "R4411-6")
    assert rows[1].identifiers == ()
    assert [row.has_warning for row in rows] == [False, True]
    assert (rows[0].page_start, rows[1].page_end) == (3, 4)
    assert all(
        row.content == chunk.content for row, chunk in zip(rows, prepared.chunks, strict=True)
    )


def test_embedding_dimension() -> None:
    _, rows = build_rows(_prepared(), FakeEmbedder(DIMS), DIMS)
    assert all(len(row.embedding) == DIMS for row in rows)
    with pytest.raises(IngestionError):
        build_rows(_prepared(), FakeEmbedder(1024), DIMS)


def test_fake_embedder_is_deterministic_and_normalized() -> None:
    embedder = FakeEmbedder(DIMS)
    first, again, other = embedder.embed(["texte A", "texte A", "texte B"])
    assert first == again and first != other
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-9)


def test_vector_literal() -> None:
    assert db.vector_literal([0.5, -1.0, 0.25]) == "[0.5,-1.0,0.25]"


def test_sql_is_parameterized() -> None:
    for query in (db.UPSERT_DOCUMENT, db.DELETE_CHUNKS, db.INSERT_CHUNK):
        assert "{" not in query and "%s" in query
        assert not re.search(r"%[^s]", query)


# --------------------------------------------------------------------------------------------
# Écriture : idempotence et rollback
# --------------------------------------------------------------------------------------------
def test_ingestion_is_idempotent() -> None:
    database = FakeDatabase()
    first_id = ingest(_prepared(), FakeEmbedder(DIMS), database, DIMS)
    first = copy.deepcopy((database.documents, database.chunks))
    second_id = ingest(_prepared(), FakeEmbedder(DIMS), database, DIMS)
    assert first_id == second_id
    assert (database.documents, database.chunks) == first
    assert len(database.documents) == 1 and len(database.chunks[first_id]) == 2


def test_failed_insert_rolls_back_everything() -> None:
    database = FakeDatabase()
    ingest(_prepared(), FakeEmbedder(DIMS), database, DIMS)
    before = copy.deepcopy((database.documents, database.chunks))
    database.fail_on_insert = True
    with pytest.raises(RuntimeError, match="échec simulé"):
        ingest(_prepared(), FakeEmbedder(DIMS), database, DIMS)
    assert (database.documents, database.chunks) == before  # ni upsert ni DELETE conservés


# --------------------------------------------------------------------------------------------
# Empreinte du fichier
# --------------------------------------------------------------------------------------------
def test_sha256_mismatch_is_refused(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.7 contenu de test")
    actual = hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert verify_sha256(pdf, _entry(actual)) == actual
    with pytest.raises(IngestionError, match="différent du manifeste"):
        verify_sha256(pdf, _entry("0" * 64))
    with pytest.raises(IngestionError, match="absent du manifeste"):
        verify_sha256(pdf, _entry(None))
    assert verify_sha256(pdf, _entry(None), allow_missing=True) == actual  # DRY-RUN seulement
    with pytest.raises(IngestionError, match="fichier absent"):
        verify_sha256(tmp_path / "absent.pdf", _entry())


# --------------------------------------------------------------------------------------------
# OpenAIEmbedder avec un faux client (aucun appel réseau)
# --------------------------------------------------------------------------------------------
class FakeEmbeddingsAPI:
    def __init__(self, dims: int, failures: int = 0) -> None:
        self.dims = dims
        self.failures = failures
        self.calls: list[int] = []

    def create(self, *, model: str, input: list[str], dimensions: int) -> SimpleNamespace:
        self.calls.append(len(input))
        if self.failures:
            self.failures -= 1
            raise openai.APIConnectionError(
                request=httpx.Request("POST", "https://example.invalid")
            )
        data = [SimpleNamespace(index=i, embedding=[0.0] * self.dims) for i in range(len(input))]
        return SimpleNamespace(data=list(reversed(data)))  # ordre rétabli par index


def _openai(api: FakeEmbeddingsAPI, sleeps: list[float]) -> OpenAIEmbedder:
    return OpenAIEmbedder(
        model="text-embedding-3-small",
        dimensions=DIMS,
        api_key="inutilisée",
        client=SimpleNamespace(embeddings=api),
        sleep=sleeps.append,
    )


def test_openai_embedder_batches_of_100() -> None:
    api = FakeEmbeddingsAPI(DIMS)
    vectors = _openai(api, []).embed([f"texte {i}" for i in range(250)])
    assert api.calls == [100, 100, 50] and len(vectors) == 250


def test_openai_embedder_retries_with_exponential_backoff() -> None:
    api, sleeps = FakeEmbeddingsAPI(DIMS, failures=2), []
    assert len(_openai(api, sleeps).embed(["un texte"])) == 1
    assert sleeps == [1.0, 2.0] and api.calls == [1, 1, 1]
    api, sleeps = FakeEmbeddingsAPI(DIMS, failures=3), []
    with pytest.raises(openai.APIConnectionError):
        _openai(api, sleeps).embed(["un texte"])
    assert len(api.calls) == 3  # 3 essais, pas un de plus


def test_openai_embedder_checks_dimensions() -> None:
    with pytest.raises(ValueError, match="1024 dimensions au lieu de 1536"):
        _openai(FakeEmbeddingsAPI(1024), []).embed(["un texte"])
