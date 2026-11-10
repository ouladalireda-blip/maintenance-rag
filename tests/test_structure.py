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


MIGRATIONS = ROOT / "supabase" / "migrations"
_SIGNATURE = re.compile(
    r"create (?:or replace )?function public\.hybrid_search\((.*?)\)\s*returns", re.S
)


def _hybrid_parameters(sql: str) -> list[str]:
    """Paramètres de hybrid_search (nom et type, sans commentaires ni espaces superflus)."""
    raw = _SIGNATURE.search(sql)
    assert raw, "hybrid_search introuvable"
    lines = [re.sub(r"--.*", "", line).strip().rstrip(",") for line in raw.group(1).splitlines()]
    return [" ".join(line.split()) for line in lines if line]


def test_later_migrations_keep_hybrid_search_signature() -> None:
    reference = _hybrid_parameters((MIGRATIONS / "0001_init.sql").read_text(encoding="utf-8"))
    for name in ("0002_chunks_warning.sql", "0003_fulltext_french.sql"):
        sql = (MIGRATIONS / name).read_text(encoding="utf-8")
        assert _hybrid_parameters(sql) == reference, name
        assert sql.lstrip().splitlines()[-1] == "commit;" and "\nbegin;\n" in sql, name
        assert "from public, anon, authenticated;" in sql and "to service_role;" in sql, name
        assert "relrowsecurity" in sql, name  # contrôle RLS


def test_fulltext_migration_avoids_double_stemming() -> None:
    sql = (MIGRATIONS / "0003_fulltext_french.sql").read_text(encoding="utf-8")
    code = "\n".join(re.sub(r"--.*", "", line) for line in sql.splitlines())
    assert "websearch_to_tsquery" not in code and "to_tsquery(" not in code
    assert "::tsquery" in code and "' | '" in code
    assert (
        "to_tsvector('public.fr_unaccent', coalesce(section_title, '') || ' ' || "
        "coalesce(content, ''))"
    ) in code


def test_total_questions_matches() -> None:
    for name in ("questions_dev.yaml", "questions_test.yaml"):
        data = _load("evals", name)
        if "total_questions" in data:
            assert data["total_questions"] == len(data["questions"]), name


def test_stopwords_migration_orders_dictionaries() -> None:
    sql = (MIGRATIONS / "0004_fr_stopwords_first.sql").read_text(encoding="utf-8")
    code = "\n".join(re.sub(r"--.*", "", line) for line in sql.splitlines())
    assert sql.lstrip().splitlines()[-1] == "commit;" and "\nbegin;\n" in sql
    assert "websearch_to_tsquery" not in code and "to_tsquery(" not in code
    mapping = re.search(r"alter mapping for [^;]*?\bwith\b([^;]*);", code, re.S)
    assert mapping, "alter mapping introuvable"
    dictionaries = [name.strip() for name in mapping.group(1).split(",")]
    assert dictionaries == ["public.fr_stopwords", "extensions.unaccent", "french_stem"]
    assert dictionaries.index("public.fr_stopwords") < dictionaries.index("extensions.unaccent")
    assert (
        "to_tsvector('public.fr_unaccent', coalesce(section_title, '') || ' ' || "
        "coalesce(content, ''))"
    ) in code  # expression identique à 0003
    assert "relrowsecurity" in code
