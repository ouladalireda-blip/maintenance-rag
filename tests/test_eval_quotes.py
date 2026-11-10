"""Chaque ``exact_quote`` de evals/questions_dev.yaml figure-t-elle dans le PDF source ?

Nécessite le PDF INRS local (jamais versionné) : SKIPPED s'il est absent, comme en CI.
Le texte du PDF est extrait en mémoire uniquement ; aucun fichier n'est écrit, et les messages
d'échec ne contiennent que des longueurs et des fragments normalisés de la CITATION, jamais du
PDF. Les échecs passent par ``pytest.fail`` pour éviter que l'introspection des assertions
n'affiche le texte du PDF.

La structure du fichier YAML est vérifiée par tests/test_structure.py::test_question_files_parse.
"""

from pathlib import Path
from typing import Any

import pymupdf
import pytest
import yaml

from maintenance_rag.normalize import normalize_for_match, quote_matches

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals" / "questions_dev.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _document(document_id: str) -> dict[str, Any]:
    documents = _load_yaml(ROOT / "corpus" / "manifest.yaml")["documents"]
    return next(document for document in documents if document["id"] == document_id)


INRS = _document("inrs-ed6109")
PDF_PATH = ROOT / INRS["path"]
PAGE_OFFSET: int = INRS["page_offset"]
MIN_CHARS: int = _load_yaml(ROOT / "config" / "config.yaml")["guardrails"]["min_quote_chars"]


def _questions_with_quotes() -> list[Any]:
    """Une entrée par question à citation ; un YAML illisible donne une seule entrée en échec
    (une erreur de collecte interromprait toute la session pytest)."""
    try:
        questions = _load_yaml(QUESTIONS)["questions"]
    except yaml.YAMLError as error:
        return [pytest.param({"yaml_error": str(error)}, id="yaml-illisible")]
    return [
        pytest.param(question, id=question["id"])
        for question in questions
        if not question["must_refuse"]
    ]


@pytest.fixture(scope="module")
def pages() -> dict[int, str]:
    """Texte normalisé de chaque page PDF (numérotée dès 1), en mémoire uniquement."""
    with pymupdf.open(PDF_PATH) as document:  # type: ignore[no-untyped-call]
        return {
            number: normalize_for_match(page.get_text())
            for number, page in enumerate(document, start=1)
        }


def _longest_prefix(needle: str, haystack: str) -> int:
    """Longueur du plus long préfixe de ``needle`` présent dans ``haystack`` (recherche
    dichotomique : si un préfixe est présent, tous les plus courts le sont aussi)."""
    low, high = 0, len(needle)
    while low < high:
        middle = (low + high + 1) // 2
        if needle[:middle] in haystack:
            low = middle
        else:
            high = middle - 1
    return low


def _diagnostic(qid: str, needle: str, haystack: str, where: str) -> str:
    prefix = _longest_prefix(needle, haystack)
    return (
        f"{qid} : citation absente {where}. Plus long préfixe trouvé : {prefix}/{len(needle)} "
        f"caractères normalisés ; suite dans la citation : {needle[prefix : prefix + 40]!r}"
    )


@pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF INRS absent (non versionné)")
@pytest.mark.parametrize("question", _questions_with_quotes())
def test_exact_quote_is_in_pdf(question: dict[str, Any], pages: dict[int, str]) -> None:
    if "yaml_error" in question:
        pytest.fail(f"{QUESTIONS.name} illisible : {question['yaml_error']}", pytrace=False)

    qid, quote = question["id"], question["exact_quote"]
    needle = normalize_for_match(quote)
    whole = "".join(pages[number] for number in sorted(pages))
    if not quote_matches(quote, whole, MIN_CHARS):
        pytest.fail(_diagnostic(qid, needle, whole, "du PDF"), pytrace=False)

    expected = question["expected_page"] + PAGE_OFFSET  # page PDF
    window = pages.get(expected, "") + pages.get(expected + 1, "")  # citation à cheval tolérée
    if not quote_matches(quote, window, MIN_CHARS):
        found = [number for number, text in pages.items() if needle in text]
        pytest.fail(
            f"{qid} : citation présente dans le PDF mais pas en page PDF {expected} "
            f"(imprimée {question['expected_page']}) ni {expected + 1} ; pages PDF où elle "
            f"figure : {found or 'à cheval sur deux pages ailleurs'}",
            pytrace=False,
        )
