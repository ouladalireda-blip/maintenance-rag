"""Tests de l'extraction des références exactes (sans PDF, sans réseau)."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from maintenance_rag.identifiers import canonical, extract_identifiers

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------
# canonical
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["R. 4411-6", "R.4411-6", "article R 4411-6", "R 4411-6"])
def test_canonical_labour_code_variants(raw: str) -> None:
    assert canonical(raw) == "R4411-6"


def test_canonical_standard() -> None:
    assert canonical("NF C18-510") == "NFC18-510"
    assert canonical("nf en iso 14118") == "NFENISO14118"


# --------------------------------------------------------------------------------------------
# Une famille par test
# --------------------------------------------------------------------------------------------
def test_standards() -> None:
    text = "Normes NF C18-510, NF EN ISO 14118, EN 60204-1 et NF C15-100 ; NF EN 12266-1."
    assert extract_identifiers(text) == [
        "EN60204-1",
        "NFC15-100",
        "NFC18-510",
        "NFEN12266-1",
        "NFENISO14118",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "l'article 536.2 de la norme NF C15-100",
        "dans la norme NF C15-100, article 536.2.",
    ],
)
def test_standard_article_both_orders(text: str) -> None:
    assert extract_identifiers(text) == ["NFC15-100", "NFC15-100:536.2"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("L'article R. 4411-6 du Code du travail", ["R4411-6"]),
        ("article R.4411-6", ["R4411-6"]),
        ("article R 4411-6", ["R4411-6"]),
        ("articles R. 4544-1 à R. 4544-11", ["R4544-1", "R4544-11"]),
        ("l'article L. 4121-1", ["L4121-1"]),
    ],
)
def test_labour_code(text: str, expected: list[str]) -> None:
    assert extract_identifiers(text) == expected


def test_eu_regulations_and_directives() -> None:
    text = "règlement européen n° 1272/2008, dit « CLP » ; directive 2006/42/CE."
    assert extract_identifiers(text) == ["1272/2008", "2006/42/CE", "CLP"]


@pytest.mark.parametrize("text", ["brochure ED 6109", "brochure ED6109"])
def test_inrs_brochures(text: str) -> None:
    assert extract_identifiers(text) == ["ED6109"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("procédé d'isolement renforcé R1", ["R1"]),
        ("Deux vannes fermées (procédé d'isolement renforcé R2)", ["R2"]),
        ("procédés d'isolement renforcé « R3 » ou R4", ["R3", "R4"]),
        ("un isolement (R1)", ["R1"]),
    ],
)
def test_isolation_processes_in_context(text: str, expected: list[str]) -> None:
    assert extract_identifiers(text) == expected


# --------------------------------------------------------------------------------------------
# Cas négatifs
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "voir page 12",
        "figure 12",
        "édition 2020",
        "la vanne R1 est ouverte",  # R1 hors contexte « procédé » / « isolement »
        "prix : 1 250,00 €",
        "le 10/11/2026",
        "réalisé en 2015 et en 2016",
        "procédé fiable. La vanne R1 reste fermée",  # le contexte s'arrête au point
    ],
)
def test_no_false_positive(text: str) -> None:
    assert extract_identifiers(text) == []


def test_unique_and_sorted() -> None:
    assert extract_identifiers("ED 6109, ED6109 et NF C18-510 puis ED 6109") == [
        "ED6109",
        "NFC18-510",
    ]


# --------------------------------------------------------------------------------------------
# Jeu d'évaluation
# --------------------------------------------------------------------------------------------
QUESTIONS = yaml.safe_load((ROOT / "evals" / "questions_dev.yaml").read_text(encoding="utf-8"))
VALID = [q for q in QUESTIONS["questions"] if not q["must_refuse"]]


@pytest.mark.parametrize("question", VALID, ids=[q["id"] for q in VALID])
def test_expected_identifiers_are_extracted(question: dict[str, Any]) -> None:
    expected = {canonical(raw) for raw in question["expected_identifiers"]}
    extracted = set(extract_identifiers(question["question"])) | set(
        extract_identifiers(question["exact_quote"])
    )
    missing = sorted(expected - extracted)
    assert not missing, f"{question['id']} : non extraits de la question ni de la citation"
