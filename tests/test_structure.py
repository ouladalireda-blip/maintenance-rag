"""Tests structurels : cohérence des fichiers de référence (sans réseau, sans clé, sans PDF)."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(*parts: str) -> dict:
    return yaml.safe_load(ROOT.joinpath(*parts).read_text(encoding="utf-8"))


def test_package_importable() -> None:
    import maintenance_rag

    assert maintenance_rag.__doc__


def test_embedding_dimension_matches_sql_schema() -> None:
    config = _load("config", "config.yaml")
    sql = (ROOT / "supabase" / "migrations" / "0001_init.sql").read_text(encoding="utf-8")
    dims = [int(d) for d in re.findall(r"vector\((\d+)\)", sql)]
    # Deux occurrences attendues : colonne chunks.embedding et paramètre de hybrid_search.
    assert dims == [config["embeddings"]["dimensions"]] * 2


def test_question_files_parse() -> None:
    for name in ("questions_dev.yaml", "questions_test.yaml"):
        assert isinstance(_load("evals", name)["questions"], list)


def test_manifest_ids_unique() -> None:
    ids = [doc["id"] for doc in _load("corpus", "manifest.yaml")["documents"]]
    assert len(ids) == len(set(ids))  # id = documents.slug, unique en base


def test_manifest_page_offsets_valid() -> None:
    for doc in _load("corpus", "manifest.yaml")["documents"]:
        assert isinstance(doc["page_offset"], int) and doc["page_offset"] >= 0
