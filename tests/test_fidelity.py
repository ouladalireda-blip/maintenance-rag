"""Tests de evals/fidelity.py (relecture humaine de la fidélité des résumés), sur des runs et
des questions fabriqués : aucun réseau, aucun vrai run, jamais le jeu test."""

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from evals.fidelity import (
    FidelityError,
    Item,
    check_dev_run,
    latest_dev_run,
    load_items,
    merge_verdicts,
    render_review,
    summarize,
    validate_verdicts,
)

QUOTE = "Le délai d'attente avant intervention est de 5 minutes après la coupure"


def _record(qid: str, status: str = "ANSWERED", must_refuse: bool = False) -> dict[str, Any]:
    return {
        "id": qid,
        "question": f"Question {qid} ?",
        "category": "near_miss" if must_refuse else "valid",
        "must_refuse": must_refuse,
        "status": status,
        "reponse": f"Résumé {qid}.",
        "prompt_version": "generation-v2-2026-11-11",
        "citations": [
            {
                "passage_id": "P1",
                "chunk_id": 1,
                "section_title": "5.1. Consignation électrique",
                "printed_page_start": 14,
                "printed_page_end": 15,
                "texte": "Affirmation.",
                "extrait_exact": QUOTE,
            }
        ],
    }


@pytest.fixture
def dev_files(tmp_path: Path) -> tuple[Path, Path]:
    questions = {
        "questions": [
            {"id": f"INRS-DEV-00{i}", "exact_quote": f"Citation attendue {i}.", "expected_page": i}
            for i in range(1, 5)
        ]
    }
    questions_path = tmp_path / "questions_dev.yaml"
    questions_path.write_text(yaml.safe_dump(questions, allow_unicode=True), encoding="utf-8")
    records = [
        _record("INRS-DEV-001"),
        _record("INRS-DEV-002", status="REFUSED_BY_GUARDRAIL"),  # valide refusée : exclue
        _record("INRS-DEV-003", status="ANSWERED", must_refuse=True),  # piège : exclu
        _record("INRS-DEV-004"),
    ]
    run_path = tmp_path / "e2e_dev_2026-11-11_abc_run1.jsonl"
    run_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    return run_path, questions_path


def test_only_valid_answered_questions_are_reviewed(dev_files: tuple[Path, Path]) -> None:
    run_path, questions_path = dev_files
    items, version = load_items(run_path, questions_path)
    assert [item.id for item in items] == ["INRS-DEV-001", "INRS-DEV-004"]
    assert version == "generation-v2-2026-11-11"
    assert items[0].expected_quote == "Citation attendue 1." and items[0].expected_page == 1
    assert items[0].summary == "Résumé INRS-DEV-001."


def test_review_contains_summary_quotes_and_expected_quote(dev_files: tuple[Path, Path]) -> None:
    run_path, questions_path = dev_files
    items, version = load_items(run_path, questions_path)
    text = render_review(items, run_path, version)
    assert "## INRS-DEV-001" in text and "## INRS-DEV-004" in text
    assert "INRS-DEV-002" not in text and "INRS-DEV-003" not in text
    assert "> Résumé INRS-DEV-001." in text
    assert f"   > {QUOTE}" in text
    assert "(p. 14-15, 5.1. Consignation électrique)" in text
    assert "**Citation attendue** (p. 1) :" in text and "> Citation attendue 1." in text
    assert "non versionné" in text


def test_only_dev_runs_are_accepted(tmp_path: Path) -> None:
    with pytest.raises(FidelityError):
        check_dev_run(tmp_path / "e2e_test_2026-11-11_abc_run1.jsonl")
    check_dev_run(tmp_path / "e2e_dev_2026-11-11_abc_run1.jsonl")


def test_latest_dev_run_ignores_other_files(tmp_path: Path) -> None:
    older = tmp_path / "e2e_dev_2026-11-11_a.jsonl"
    newer = tmp_path / "e2e_dev_2026-11-11_b_run3.jsonl"
    test_run = tmp_path / "e2e_test_2026-11-12_c_run1.jsonl"  # plus récent, mais jeu test
    for index, path in enumerate((older, newer, test_run)):
        path.write_text("", encoding="utf-8")
        os.utime(path, (1_000_000 + index, 1_000_000 + index))
    assert latest_dev_run(tmp_path) == newer
    with pytest.raises(FidelityError):
        latest_dev_run(tmp_path / "vide")


def _items(*ids: str) -> list[Item]:
    return [Item(qid, "q", "r", [], "c", 1) for qid in ids]


def test_merge_keeps_existing_verdicts() -> None:
    existing = {
        "questions": [
            {"id": "INRS-DEV-001", "verdict": "trompeur", "note": "généralise au seul neutre"},
            {"id": "INRS-DEV-009", "verdict": "fidele", "note": ""},  # plus ANSWERED
        ]
    }
    merged, stale = merge_verdicts(existing, _items("INRS-DEV-001", "INRS-DEV-004"), "run", "v2")
    rows = {row["id"]: row for row in merged["questions"]}
    assert rows["INRS-DEV-001"] == {
        "id": "INRS-DEV-001",
        "verdict": "trompeur",
        "note": "généralise au seul neutre",
    }
    assert rows["INRS-DEV-004"] == {"id": "INRS-DEV-004", "verdict": None, "note": ""}
    assert rows["INRS-DEV-009"]["verdict"] == "fidele"  # aucun verdict perdu
    assert stale == ["INRS-DEV-009"]
    assert merged["run"] == "run" and merged["set"] == "dev"


def test_merged_file_has_no_guide_text(dev_files: tuple[Path, Path]) -> None:
    run_path, questions_path = dev_files
    items, version = load_items(run_path, questions_path)
    merged, _ = merge_verdicts({"questions": []}, items, run_path.name, version)
    dumped = yaml.safe_dump(merged, allow_unicode=True)
    assert QUOTE not in dumped and "Résumé" not in dumped and "Citation attendue" not in dumped
    assert set(merged["questions"][0]) == {"id", "verdict", "note"}


@pytest.mark.parametrize(
    "row",
    [
        {"id": "INRS-DEV-001", "verdict": "bof", "note": ""},
        {"id": "INRS-TEST-001", "verdict": None, "note": ""},  # jamais le jeu test
        {"id": "INRS-DEV-001", "verdict": "fidele", "note": "x" * 201},
        {"id": "INRS-DEV-001", "verdict": "trompeur", "note": f"il dit : {QUOTE}"},
    ],
    ids=["verdict-inconnu", "jeu-test", "note-longue", "note-recopie-le-guide"],
)
def test_invalid_verdicts_are_refused(row: dict[str, Any]) -> None:
    with pytest.raises(FidelityError):
        validate_verdicts({"questions": [row]}, guide_texts=[QUOTE])


def test_duplicate_ids_are_refused() -> None:
    rows = [{"id": "INRS-DEV-001", "verdict": None, "note": ""}] * 2
    with pytest.raises(FidelityError):
        validate_verdicts({"questions": rows})


def test_valid_verdicts_pass() -> None:
    rows = [
        {"id": "INRS-DEV-001", "verdict": "fidele", "note": ""},
        {
            "id": "INRS-DEV-002",
            "verdict": "trompeur",
            "note": "réduit « tous les conducteurs » au neutre",
        },
        {"id": "INRS-DEV-003", "verdict": None, "note": ""},
    ]
    validate_verdicts({"questions": rows}, guide_texts=[QUOTE])


def test_summary_counts_rates_and_misleading() -> None:
    verdicts = ["fidele"] * 17 + ["incomplet"] * 2 + ["trompeur"] * 1
    rows = [
        {"id": f"INRS-DEV-{i:03d}", "verdict": verdict, "note": ""}
        for i, verdict in enumerate(verdicts, start=1)
    ]
    result = summarize({"questions": rows})
    assert result["total"] == 20 and result["reviewed"] == 20 and result["pending"] == []
    assert result["counts"] == {"fidele": 17, "incomplet": 2, "trompeur": 1}
    assert result["rate_of_reviewed"]["trompeur"] == pytest.approx(0.05)
    assert result["misleading"] == ["INRS-DEV-020"]
    assert result["misleading_ok"] is True  # 1/20 = 5 % : seuil atteint pile
    rows.append({"id": "INRS-DEV-021", "verdict": "trompeur", "note": ""})
    assert summarize({"questions": rows})["misleading_ok"] is False  # 2/21 > 5 %


def test_summary_with_pending_reviews() -> None:
    rows = [
        {"id": "INRS-DEV-001", "verdict": "fidele", "note": ""},
        {"id": "INRS-DEV-002", "verdict": None, "note": ""},
    ]
    result = summarize({"questions": rows})
    assert result["reviewed"] == 1 and result["pending"] == ["INRS-DEV-002"]
    assert result["rate_of_reviewed"]["fidele"] == 1.0
    assert result["rate_of_total"]["fidele"] == 0.5
