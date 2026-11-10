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


CATEGORIES = {"valid", "near_miss", "identifier_trap", "out_of_scope"}
REFUSAL_NULL_FIELDS = ("target_document_id", "expected_section", "expected_page", "exact_quote")


def test_question_files_parse() -> None:
    """Les fichiers de questions se chargent et chaque question respecte le format.

    Les règles détaillées ne s'appliquent qu'aux fichiers déjà remplis (questions_test.yaml
    est encore vide tant qu'il n'est pas rédigé et figé).
    """
    for name in ("questions_dev.yaml", "questions_test.yaml"):
        raw = (ROOT / "evals" / name).read_text(encoding="utf-8")
        assert "À VÉRIFIER" not in raw, f"{name} : commentaire « À VÉRIFIER » restant"
        data = yaml.safe_load(raw)
        questions = data["questions"]
        assert isinstance(questions, list)
        if not questions:
            continue
        assert data["page_convention"] == "printed", name
        ids = [question["id"] for question in questions]
        assert len(ids) == len(set(ids)), f"{name} : ids en double"
        for question in questions:
            qid = question["id"]
            assert question["category"] in CATEGORIES, qid
            assert isinstance(question["must_refuse"], bool), qid
            if question["must_refuse"]:
                for key in REFUSAL_NULL_FIELDS:
                    assert question[key] is None, f"{qid} : {key} doit être null"
                assert question["expected_identifiers"] == [], qid
            else:
                assert question["target_document_id"] == "inrs-ed6109", qid
                page = question["expected_page"]
                assert isinstance(page, int) and not isinstance(page, bool), qid
                quote = question["exact_quote"]
                assert isinstance(quote, str) and quote.strip(), f"{qid} : exact_quote vide"


def test_manifest_ids_unique() -> None:
    ids = [doc["id"] for doc in _load("corpus", "manifest.yaml")["documents"]]
    assert len(ids) == len(set(ids))  # id = documents.slug, unique en base


def test_manifest_page_offsets_valid() -> None:
    for doc in _load("corpus", "manifest.yaml")["documents"]:
        assert isinstance(doc["page_offset"], int) and doc["page_offset"] >= 0
