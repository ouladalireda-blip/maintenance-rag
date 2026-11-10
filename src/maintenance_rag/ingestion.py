"""Ingestion du corpus : lecture du manifeste, vérification du SHA-256, extraction du texte PDF
page par page (PyMuPDF) avec numéros de page imprimés.

Pipeline d'un document :
    manifeste → SHA-256 du PDF (refus s'il diffère) → read_pdf → chunk_blocks (max_tokens de
    config.yaml) → identifiants de chaque passage (sur ``content``) → embeddings de
    ``embedding_text`` → écriture en une transaction (db.write_document).

Usage :
    uv run python -m maintenance_rag.ingestion --doc inrs-ed6109            # DRY-RUN
    uv run python -m maintenance_rag.ingestion --doc inrs-ed6109 --apply    # écriture réelle

Le DRY-RUN (par défaut) fait tout sauf l'appel d'embedding et l'écriture : il ne lit pas .env,
n'ouvre aucune connexion et n'affiche que des nombres et des identifiants (références), jamais
de texte du PDF.
"""

import argparse
import hashlib
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from maintenance_rag.chunking import chunk_blocks
from maintenance_rag.config import (
    CONFIG_PATH,
    ROOT,
    LayoutProfile,
    ManifestEntry,
    load_layout_profile,
    load_manifest,
)
from maintenance_rag.db import ChunkRow, Connection, DocumentRow, write_document
from maintenance_rag.embeddings import Embedder
from maintenance_rag.identifiers import extract_identifiers
from maintenance_rag.pdf_reader import read_pdf
from maintenance_rag.schemas import Chunk


class IngestionError(RuntimeError):
    """Ingestion refusée (empreinte, manifeste ou configuration incohérents)."""


@dataclass(frozen=True)
class Prepared:
    """Document prêt à être vectorisé et écrit."""

    entry: ManifestEntry
    sha256: str
    chunks: list[Chunk]
    identifiers: list[tuple[str, ...]]  # par passage, même ordre que ``chunks``


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, entry: ManifestEntry, *, allow_missing: bool = False) -> str:
    """Empreinte du fichier ; refus si elle diffère du manifeste.

    Une empreinte absente du manifeste (``null``) est refusée, sauf ``allow_missing`` (DRY-RUN
    uniquement : l'empreinte calculée est alors affichée pour être reportée au manifeste).
    """
    if not path.exists():
        raise IngestionError(f"{entry.id} : fichier absent ({entry.path})")
    actual = file_sha256(path)
    if entry.sha256 is None:
        if not allow_missing:
            raise IngestionError(f"{entry.id} : sha256 absent du manifeste, ingestion refusée")
        return actual
    if actual != entry.sha256.lower():
        raise IngestionError(
            f"{entry.id} : sha256 du fichier ({actual}) différent du manifeste "
            f"({entry.sha256}) : autre édition du document, ingestion refusée"
        )
    return actual


def prepare(
    entry: ManifestEntry,
    profile: LayoutProfile,
    max_tokens: int,
    *,
    root: Path = ROOT,
    allow_missing_sha: bool = False,
) -> Prepared:
    path = root / entry.path
    sha256 = verify_sha256(path, entry, allow_missing=allow_missing_sha)
    chunks = chunk_blocks(read_pdf(path, entry, profile), max_tokens, document_slug=entry.id)
    identifiers = [tuple(extract_identifiers(chunk.content)) for chunk in chunks]
    return Prepared(entry=entry, sha256=sha256, chunks=chunks, identifiers=identifiers)


def document_row(entry: ManifestEntry, sha256: str) -> DocumentRow:
    return DocumentRow(
        slug=entry.id,
        title=entry.title,
        source_url=entry.source_url,
        license=entry.license,
        language=entry.language,
        version=entry.version,
        machine=entry.machine,
        page_offset=entry.page_offset,
        sha256=sha256,
    )


def chunk_rows(
    chunks: Sequence[Chunk],
    identifiers: Sequence[tuple[str, ...]],
    vectors: Sequence[Sequence[float]],
    dimensions: int,
) -> list[ChunkRow]:
    if not len(chunks) == len(identifiers) == len(vectors):
        raise IngestionError("nombre de passages, d'identifiants et de vecteurs différents")
    rows = []
    for chunk, ids, vector in zip(chunks, identifiers, vectors, strict=True):
        if len(vector) != dimensions:
            raise IngestionError(f"passage {chunk.chunk_index} : vecteur de {len(vector)} dims")
        rows.append(
            ChunkRow(
                chunk_index=chunk.chunk_index,
                section_title=chunk.section_title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                content=chunk.content,
                token_count=chunk.token_count,
                identifiers=ids,
                embedding=tuple(vector),
                has_warning=chunk.has_warning,
            )
        )
    return rows


def build_rows(
    prepared: Prepared, embedder: Embedder, dimensions: int
) -> tuple[DocumentRow, list[ChunkRow]]:
    """Vectorise ``embedding_text`` de chaque passage et construit les lignes à écrire."""
    if embedder.dimensions != dimensions:
        raise IngestionError(
            f"embedder de dimension {embedder.dimensions}, config.yaml : {dimensions}"
        )
    vectors = embedder.embed([chunk.embedding_text for chunk in prepared.chunks])
    rows = chunk_rows(prepared.chunks, prepared.identifiers, vectors, dimensions)
    return document_row(prepared.entry, prepared.sha256), rows


def ingest(prepared: Prepared, embedder: Embedder, connection: Connection, dimensions: int) -> Any:
    """Vectorise puis écrit le document en une transaction ; renvoie l'id du document."""
    document, rows = build_rows(prepared, embedder, dimensions)
    return write_document(connection, document, rows)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
class Settings(BaseSettings):
    """Secrets lus dans .env ou l'environnement, UNIQUEMENT avec --apply."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    supabase_db_url: SecretStr
    openai_api_key: SecretStr


def _ranges(numbers: Iterable[int]) -> str:
    values = sorted(set(numbers))
    if not values:
        return "aucune"
    parts: list[str] = []
    start = previous = values[0]
    for value in [*values[1:], None]:
        if value is not None and value == previous + 1:
            previous = value
            continue
        parts.append(f"{start}" if start == previous else f"{start}-{previous}")
        if value is not None:
            start = previous = value
    return ", ".join(parts)


def summarize(prepared: Prepared, sha_in_manifest: bool) -> None:
    """Bilan du DRY-RUN : nombres et références uniquement."""
    chunks = prepared.chunks
    distinct = sorted({ident for ids in prepared.identifiers for ident in ids})
    pdf_pages = [p for c in chunks for p in range(c.page_start, c.page_end + 1)]
    printed = [
        p
        for c in chunks
        if c.printed_page_start is not None and c.printed_page_end is not None
        for p in range(c.printed_page_start, c.printed_page_end + 1)
    ]
    print(f"Document                  : {prepared.entry.id}")
    status = "conforme au manifeste" if sha_in_manifest else "ABSENT du manifeste"
    print(f"sha256 du fichier         : {prepared.sha256} ({status})")
    print(f"Passages                  : {len(chunks)}")
    print(f"Tokens à vectoriser       : {sum(c.token_count for c in chunks)}")
    print(f"Passages avec identifiants: {sum(1 for ids in prepared.identifiers if ids)}")
    print(f"Identifiants distincts    : {len(distinct)}")
    for ident in distinct:
        print(f"  - {ident}")
    print(f"Passages has_warning      : {sum(c.has_warning for c in chunks)}")
    print(f"Pages PDF couvertes       : {_ranges(pdf_pages)}")
    print(f"Pages imprimées couvertes : {_ranges(printed)}")


def main(argv: Sequence[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="Ingestion d'un document du corpus.")
    parser.add_argument("--doc", required=True, help="id du document dans corpus/manifest.yaml")
    parser.add_argument(
        "--apply", action="store_true", help="vectoriser et écrire en base (sinon DRY-RUN)"
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    entries = {entry.id: entry for entry in load_manifest()}
    if args.doc not in entries:
        parser.error(f"document inconnu du manifeste : {args.doc}")
    entry = entries[args.doc]
    if entry.layout_profile is None:
        parser.error(f"{entry.id} : layout_profile absent du manifeste")
    profile = load_layout_profile(entry.layout_profile)
    max_tokens = int(config["chunking"]["max_tokens"])
    embeddings = config["embeddings"]

    try:
        prepared = prepare(entry, profile, max_tokens, allow_missing_sha=not args.apply)
    except IngestionError as error:
        print(f"REFUS : {error}")
        return 1

    if not args.apply:
        print("DRY-RUN : aucun appel d'embedding, aucune écriture en base, .env non lu.\n")
        summarize(prepared, sha_in_manifest=entry.sha256 is not None)
        if entry.sha256 is None:
            print("\nATTENTION : reporter ce sha256 dans corpus/manifest.yaml avant --apply.")
        return 0

    import psycopg

    from maintenance_rag.embeddings import OpenAIEmbedder

    settings = Settings()  # valeurs lues dans .env
    embedder = OpenAIEmbedder(
        model=embeddings["model"],
        dimensions=int(embeddings["dimensions"]),
        api_key=settings.openai_api_key.get_secret_value(),
        batch_size=int(embeddings["batch_size"]),
        max_attempts=int(embeddings["max_attempts"]),
    )
    with psycopg.connect(settings.supabase_db_url.get_secret_value()) as connection:
        document_id = ingest(prepared, embedder, connection, int(embeddings["dimensions"]))
    print(f"Écrit : document {entry.id} ({document_id}), {len(prepared.chunks)} passages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
