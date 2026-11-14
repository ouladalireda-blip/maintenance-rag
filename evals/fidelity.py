"""Relecture humaine de la FIDÉLITÉ des résumés du LLM (jeu dev ; protocole : ADR 0003).

Le garde-fou 2 vérifie que chaque extrait cité existe mot pour mot, pas que le résumé
(``reponse``) dit la même chose que l'extrait. Constat INRS-DEV-028 : bonne source, extrait
exact, résumé trompeur. Ce script outille une relecture À LA MAIN :

    uv run python evals/fidelity.py review [--run evals/runs/<fichier>.jsonl]
    uv run python evals/fidelity.py summary

- ``review`` : à partir du dernier run e2e du jeu DEV (``evals/runs/e2e_dev_*.jsonl``, le plus
  récent, sauf ``--run``), écrit
    * ``evals/runs/fidelity_review_dev.md`` (NON versionné : contient des extraits du guide) :
      pour chaque question valide ANSWERED, l'id, la question, le résumé du LLM, les extraits
      cités (page, section) et la citation attendue ;
    * ``evals/fidelity_dev.yaml`` (versionné, sans texte du guide) : une ligne par question,
      ``verdict: null`` à remplir (fidele | incomplet | trompeur) et ``note: ""``. Les verdicts
      déjà remplis sont conservés ; une question qui n'est plus ANSWERED dans ce run est gardée
      et signalée.
- ``summary`` : nombre et taux de chaque verdict (sur les questions relues et sur le total),
  questions non relues, liste des « trompeur », et rappel du critère de l'ADR 0003
  (trompeurs ≤ 5 % des réponses).

Le jeu test n'est jamais lu : seuls les runs ``e2e_dev_*`` sont acceptés.
"""

import argparse
import datetime as dt
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "evals" / "runs"
QUESTIONS = ROOT / "evals" / "questions_dev.yaml"
REVIEW = RUNS / "fidelity_review_dev.md"
VERDICTS = ROOT / "evals" / "fidelity_dev.yaml"
VERDICT_VALUES = ("fidele", "incomplet", "trompeur")
MAX_MISLEADING_RATE = 0.05  # ADR 0003 : résumés trompeurs ≤ 5 % des réponses
NOTE_MAX_CHARS = 200
GUIDE_FRAGMENT_CHARS = 30  # une note ne doit contenir aucun fragment du guide de cette longueur


class FidelityError(ValueError):
    """Fichier ou verdict invalide (message lisible)."""


@dataclass(frozen=True)
class Item:
    id: str
    question: str
    summary: str
    citations: list[dict[str, Any]]
    expected_quote: str
    expected_page: int


# ----------------------------------------------------------------------------- lecture du run
def latest_dev_run(runs_dir: Path = RUNS) -> Path:
    """Run e2e du jeu DEV le plus récent (date de modification)."""
    candidates = [p for p in runs_dir.glob("e2e_dev_*.jsonl") if p.is_file()]
    if not candidates:
        raise FidelityError(f"aucun run e2e du jeu dev dans {runs_dir}")
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def check_dev_run(path: Path) -> None:
    if not path.name.startswith("e2e_dev_"):
        raise FidelityError(f"{path.name} : seuls les runs du jeu DEV (e2e_dev_*) sont acceptés")


def load_items(run_path: Path, questions_path: Path = QUESTIONS) -> tuple[list[Item], str]:
    """Questions valides ANSWERED du run, avec leur citation attendue ; et la PROMPT_VERSION."""
    check_dev_run(run_path)
    expected = {
        q["id"]: q for q in yaml.safe_load(questions_path.read_text(encoding="utf-8"))["questions"]
    }
    items: list[Item] = []
    versions: set[str] = set()
    for line in run_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        versions.add(str(record.get("prompt_version")))
        if record["must_refuse"] or record["status"] != "ANSWERED":
            continue
        question = expected[record["id"]]
        items.append(
            Item(
                id=record["id"],
                question=record["question"],
                summary=record["reponse"],
                citations=list(record["citations"]),
                expected_quote=question["exact_quote"],
                expected_page=int(question["expected_page"]),
            )
        )
    return items, ", ".join(sorted(versions))


# ----------------------------------------------------------------------------- relecture
def _pages(citation: dict[str, Any]) -> str:
    start, end = citation.get("printed_page_start"), citation.get("printed_page_end")
    if start is None:
        return "page non numérotée"
    return f"p. {start}" if end in (None, start) else f"p. {start}-{end}"


def render_review(items: Sequence[Item], run_path: Path, prompt_version: str) -> str:
    lines = [
        "# Relecture de la fidélité des résumés — jeu dev",
        "",
        "Fichier LOCAL, non versionné : il contient des extraits du guide INRS ED 6109.",
        "",
        f"- Run : `{run_path.name}` (prompt {prompt_version})",
        f"- Questions valides ANSWERED : {len(items)}",
        "- Verdicts à saisir dans `evals/fidelity_dev.yaml` : `fidele` (le résumé dit ce que dit "
        "l'extrait), `incomplet` (juste mais partiel), `trompeur` (le résumé déforme l'extrait ou "
        "ajoute une affirmation qu'il ne contient pas). Les notes ne doivent contenir aucun texte "
        "du guide.",
        "",
    ]
    for item in items:
        lines += [
            f"## {item.id}",
            "",
            f"**Question :** {item.question}",
            "",
            "**Résumé du LLM :**",
            "",
            f"> {item.summary}",
            "",
            "**Extraits cités :**",
            "",
        ]
        for number, citation in enumerate(item.citations, start=1):
            section = citation.get("section_title") or "sans section"
            lines += [
                f"{number}. ({_pages(citation)}, {section})",
                "",
                f"   > {citation['extrait_exact']}",
                "",
            ]
        lines += [
            f"**Citation attendue** (p. {item.expected_page}) :",
            "",
            f"> {item.expected_quote}",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- verdicts
def load_verdicts(path: Path = VERDICTS) -> dict[str, Any]:
    if not path.exists():
        return {"questions": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("questions", [])
    return dict(data)


def guide_fragments(texts: Sequence[str], size: int = GUIDE_FRAGMENT_CHARS) -> set[str]:
    """Tous les fragments normalisés de ``size`` caractères des textes du guide."""
    from maintenance_rag.normalize import normalize_for_match

    fragments: set[str] = set()
    for text in texts:
        norm = normalize_for_match(text)
        fragments.update(norm[i : i + size] for i in range(max(0, len(norm) - size + 1)))
    return fragments


def validate_verdicts(data: dict[str, Any], guide_texts: Sequence[str] = ()) -> None:
    """Verdicts autorisés, ids uniques, notes courtes et sans fragment du guide."""
    from maintenance_rag.normalize import normalize_for_match

    fragments = guide_fragments(guide_texts)
    seen: set[str] = set()
    for row in data["questions"]:
        qid = row.get("id")
        if not isinstance(qid, str) or not qid.startswith("INRS-DEV-"):
            raise FidelityError(f"id invalide : {qid!r} (jeu dev uniquement)")
        if qid in seen:
            raise FidelityError(f"{qid} : en double")
        seen.add(qid)
        if row.get("verdict") not in (None, *VERDICT_VALUES):
            raise FidelityError(
                f"{qid} : verdict {row.get('verdict')!r} ; attendu : {', '.join(VERDICT_VALUES)}"
            )
        note = row.get("note") or ""
        if not isinstance(note, str) or len(note) > NOTE_MAX_CHARS:
            raise FidelityError(f"{qid} : note trop longue (≤ {NOTE_MAX_CHARS} caractères)")
        norm = normalize_for_match(note)
        if any(
            norm[i : i + GUIDE_FRAGMENT_CHARS] in fragments
            for i in range(max(0, len(norm) - GUIDE_FRAGMENT_CHARS + 1))
        ):
            raise FidelityError(f"{qid} : la note recopie du texte du guide")


def merge_verdicts(
    existing: dict[str, Any], items: Sequence[Item], run_name: str, prompt_version: str
) -> tuple[dict[str, Any], list[str]]:
    """Une ligne par question ANSWERED du run ; verdicts et notes existants conservés.

    Renvoie aussi les ids déjà présents mais qui ne sont plus ANSWERED dans ce run (gardés,
    pour ne perdre aucun verdict, et signalés)."""
    previous = {row["id"]: row for row in existing.get("questions", [])}
    current = [item.id for item in items]
    rows = [
        {
            "id": qid,
            "verdict": previous.get(qid, {}).get("verdict"),
            "note": previous.get(qid, {}).get("note") or "",
        }
        for qid in current
    ]
    stale = [qid for qid in previous if qid not in set(current)]
    rows += [previous[qid] for qid in stale]
    rows.sort(key=lambda row: row["id"])
    merged = {
        "set": "dev",
        "run": run_name,
        "prompt_version": prompt_version,
        "updated": dt.date.today().isoformat(),
        "verdict_values": list(VERDICT_VALUES),
        "questions": rows,
    }
    return merged, stale


def dump_verdicts(data: dict[str, Any], path: Path = VERDICTS) -> None:
    header = (
        "# Fidélité des résumés du LLM, jeu DEV : verdicts saisis À LA MAIN (ADR 0003).\n"
        "# verdict : fidele | incomplet | trompeur (null = pas encore relu).\n"
        "# note : courte, SANS texte du guide. Relecture : evals/runs/fidelity_review_dev.md\n"
        "# (local). Régénérer : uv run python evals/fidelity.py review\n"
    )
    body = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100)
    path.write_text(header + body, encoding="utf-8")


def review(run_path: Path | None = None) -> int:
    run_path = run_path or latest_dev_run()
    items, prompt_version = load_items(run_path)
    RUNS.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text(render_review(items, run_path, prompt_version), encoding="utf-8")
    existing = load_verdicts()
    guide_texts = [c["extrait_exact"] for item in items for c in item.citations] + [
        item.expected_quote for item in items
    ]
    validate_verdicts(existing, guide_texts)  # aucune note ne doit recopier le guide
    merged, stale = merge_verdicts(existing, items, run_path.name, prompt_version)
    dump_verdicts(merged)
    filled = sum(1 for row in merged["questions"] if row["verdict"] is not None)
    print(f"Run : {run_path.relative_to(ROOT).as_posix()} (prompt {prompt_version})")
    print(f"Questions valides ANSWERED : {len(items)}")
    print(f"Relecture (locale, non versionnée) : {REVIEW.relative_to(ROOT).as_posix()}")
    print(
        f"Verdicts (versionné) : {VERDICTS.relative_to(ROOT).as_posix()} — "
        f"{filled}/{len(merged['questions'])} déjà saisis"
    )
    if stale:
        print(f"Gardées mais plus ANSWERED dans ce run : {', '.join(stale)}")
    return 0


# ----------------------------------------------------------------------------- synthèse
def summarize(data: dict[str, Any]) -> dict[str, Any]:
    rows = data["questions"]
    total = len(rows)
    reviewed = [row for row in rows if row.get("verdict") is not None]
    counts = {
        value: sum(1 for row in reviewed if row["verdict"] == value) for value in VERDICT_VALUES
    }
    misleading = [row["id"] for row in reviewed if row["verdict"] == "trompeur"]
    return {
        "total": total,
        "reviewed": len(reviewed),
        "pending": [row["id"] for row in rows if row.get("verdict") is None],
        "counts": counts,
        "rate_of_reviewed": {
            value: (count / len(reviewed) if reviewed else None) for value, count in counts.items()
        },
        "rate_of_total": {
            value: (count / total if total else None) for value, count in counts.items()
        },
        "misleading": misleading,
        "misleading_ok": (len(misleading) / total <= MAX_MISLEADING_RATE) if total else None,
    }


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def summary(path: Path = VERDICTS) -> int:
    data = load_verdicts(path)
    validate_verdicts(data)
    result = summarize(data)
    print(
        f"Fidélité des résumés — {data.get('run', '?')} (prompt {data.get('prompt_version', '?')})"
    )
    print(f"Questions valides ANSWERED : {result['total']} ; relues : {result['reviewed']}")
    print(f"{'verdict':10} {'nombre':>6} {'% relues':>9} {'% total':>8}")
    for value in VERDICT_VALUES:
        print(
            f"{value:10} {result['counts'][value]:>6} {_pct(result['rate_of_reviewed'][value]):>9} "
            f"{_pct(result['rate_of_total'][value]):>8}"
        )
    print(
        f"Non relues : {len(result['pending'])}"
        + (f" ({', '.join(result['pending'])})" if result["pending"] else "")
    )
    print(f"Trompeurs : {', '.join(result['misleading']) or 'aucun'}")
    if result["pending"]:
        print("Critère (trompeurs ≤ 5 % des réponses) : non évaluable tant que tout n'est pas relu")
    elif result["misleading_ok"] is not None:
        verdict = "OK" if result["misleading_ok"] else "ÉCHEC"
        print(f"Critère (trompeurs ≤ 5 % des réponses) : {verdict}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="Relecture de la fidélité des résumés (dev).")
    sub = parser.add_subparsers(dest="command", required=True)
    review_parser = sub.add_parser("review", help="générer la relecture et le fichier de verdicts")
    review_parser.add_argument("--run", type=Path, default=None, help="run e2e dev (.jsonl)")
    sub.add_parser("summary", help="synthèse des verdicts saisis")
    args = parser.parse_args(argv)
    try:
        return review(args.run) if args.command == "review" else summary()
    except FidelityError as error:
        print(f"ERREUR : {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
