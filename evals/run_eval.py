"""Lancement d'une évaluation sur questions_dev.yaml ou questions_test.yaml ; écrit les sorties
dans evals/results/ (versionné : identifiants et chiffres uniquement, jamais de texte).

Mode « retrieval » : mesure de la recherche AVANT tout reranking, branche par branche.

    uv run python evals/run_eval.py --set dev --mode retrieval

- Questions évaluées : must_refuse == false (une citation de référence existe).
- Quatre jeux de poids (semantic, full_text, identifier) passés à hybrid_search, avec
  match_count et rrf_k de config.yaml : hybride (poids de config.yaml), sémantique seule,
  full-text seul, identifiants seuls.
- Pour une configuration à une seule branche, les lignes de score RRF nul ne viennent pas de
  cette branche (elles complètent le top 10 par ordre d'id) : elles sont écartées. Si la
  branche ne renvoie rien, la question compte comme un échec (rang None).
- Diagnostic par question : un appel hybride avec un match_count très grand renvoie tous les
  candidats de chaque branche, d'où les rangs de branche du bon passage et le nombre de
  passages où la branche full-text a matché.

Base en LECTURE SEULE (connexion read_only, uniquement hybrid_search). Les vecteurs des
questions sont mis en cache dans .cache/query_embeddings.json (non versionné).
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from maintenance_rag.retrieval import CachedEmbedder, SearchResult, Weights, search
from metrics import TOP_K, first_hit_rank, hit, mrr, recall_at_k

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
CACHE = ROOT / ".cache" / "query_embeddings.json"
RESULTS = ROOT / "evals" / "results"
DIAGNOSTIC_MATCH_COUNT = 10_000  # tous les candidats de chaque branche (69 passages en base)


def _configs() -> dict[str, Weights]:
    retrieval = CONFIG["retrieval"]
    hybrid = (
        float(retrieval["semantic_weight"]),
        float(retrieval["full_text_weight"]),
        float(retrieval["identifier_weight"]),
    )
    return {
        "hybride": hybrid,
        "sémantique seule": (1.0, 0.0, 0.0),
        "full-text seul": (0.0, 1.0, 0.0),
        "identifiants seuls": (0.0, 0.0, 1.0),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _scrub(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "****")
    return text


def _error_chain(error: BaseException, secrets: Sequence[str]) -> str:
    """Types et messages de l'erreur et de ses causes, secrets masqués."""
    parts: list[str] = []
    current: BaseException | None = error
    while current is not None and len(parts) < 6:
        parts.append(f"{type(current).__module__}.{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return _scrub("\n  causé par ".join(parts), secrets)


def _fmt(value: int | None) -> str:
    return "-" if value is None else str(value)


def _evaluate(questions: list[dict[str, Any]], embedder: Any, conn: Any) -> dict[str, Any]:
    retrieval = CONFIG["retrieval"]
    match_count, rrf_k = int(retrieval["match_count"]), int(retrieval["rrf_k"])
    min_chars = int(CONFIG["guardrails"]["min_quote_chars"])
    configs = _configs()

    ranks: dict[str, dict[str, int | None]] = {name: {} for name in configs}
    returned: dict[str, dict[str, int]] = {name: {} for name in configs}
    per_question: dict[str, dict[str, Any]] = {}
    for question in questions:
        qid, text = question["id"], question["question"]
        for name, weights in configs.items():
            results = search(text, embedder, conn, weights, match_count, rrf_k=rrf_k)
            if name != "hybride":  # configuration à une branche : lignes de score nul écartées
                results = [result for result in results if result.rrf_score > 0]
            returned[name][qid] = len(results)
            ranks[name][qid] = first_hit_rank(results, question, min_chars, TOP_K)

        everything = search(
            text, embedder, conn, configs["hybride"], DIAGNOSTIC_MATCH_COUNT, rrf_k=rrf_k
        )
        gold: list[SearchResult] = [r for r in everything if hit(r, question, min_chars)]
        best = min(gold, key=lambda r: r.semantic_rank or 10**9, default=None)
        per_question[qid] = {
            "rank_hybrid": ranks["hybride"][qid],
            "gold_chunk_ids": sorted(r.chunk_id for r in gold),
            "gold_chunk_index": best.chunk_index if best else None,
            "gold_semantic_rank": best.semantic_rank if best else None,
            "gold_full_text_rank": best.full_text_rank if best else None,
            "gold_identifier_rank": best.identifier_rank if best else None,
            "full_text_returned_top10": returned["full-text seul"][qid],
            "full_text_matches_total": sum(1 for r in everything if r.full_text_rank is not None),
            "identifier_matches_total": sum(1 for r in everything if r.identifier_rank is not None),
            "chunks_in_base": len(everything),
        }

    summary = {}
    for name in configs:
        values = [ranks[name][q["id"]] for q in questions]
        summary[name] = {
            "weights": list(configs[name]),
            "recall@1": recall_at_k(values, 1),
            "recall@3": recall_at_k(values, 3),
            "recall@10": recall_at_k(values, 10),
            "mrr": mrr(values),
            "ranks": ranks[name],
            "returned": returned[name],
        }
    return {"summary": summary, "per_question": per_question}


def _print(report: dict[str, Any], questions: list[dict[str, Any]]) -> None:
    print("\na) Recherche seule (avant reranking), questions valides :", len(questions))
    print(f"{'configuration':20} {'poids (s,f,i)':15} {'R@1':>5} {'R@3':>5} {'R@10':>5} {'MRR':>6}")
    for name, row in report["summary"].items():
        weights = ",".join(f"{w:g}" for w in row["weights"])
        print(
            f"{name:20} {weights:15} {row['recall@1']:5.2f} {row['recall@3']:5.2f} "
            f"{row['recall@10']:5.2f} {row['mrr']:6.3f}"
        )

    print("\nb) Par question (config hybride) ; rangs de branche du bon passage parmi TOUS les")
    print("   candidats de la branche (appel diagnostic), « - » = absent de la branche")
    header = f"{'id':14} {'rang':>4} {'chunk':>5} {'sem':>4} {'ft':>4} {'ident':>5} {'ft top10':>8}"
    print(header)
    for qid, row in report["per_question"].items():
        print(
            f"{qid:14} {_fmt(row['rank_hybrid']):>4} {_fmt(row['gold_chunk_index']):>5} "
            f"{_fmt(row['gold_semantic_rank']):>4} {_fmt(row['gold_full_text_rank']):>4} "
            f"{_fmt(row['gold_identifier_rank']):>5} {row['full_text_returned_top10']:>8}"
        )

    print("\nc) Passages où la branche full-text a matché (websearch_to_tsquery, tous candidats)")
    print(f"{'id':14} {'ft':>4} {'ident':>5} {'en base':>7}")
    for qid, row in report["per_question"].items():
        print(
            f"{qid:14} {row['full_text_matches_total']:>4} {row['identifier_matches_total']:>5} "
            f"{row['chunks_in_base']:>7}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="Évaluation du pipeline maintenance-rag.")
    parser.add_argument("--set", choices=["dev"], required=True)
    parser.add_argument("--mode", choices=["retrieval"], required=True)
    args = parser.parse_args(argv)

    data = yaml.safe_load((ROOT / "evals" / f"questions_{args.set}.yaml").read_text("utf-8"))
    questions = [q for q in data["questions"] if not q["must_refuse"]]

    import psycopg

    from maintenance_rag.embeddings import OpenAIEmbedder
    from maintenance_rag.ingestion import Settings

    settings = Settings()  # lit .env ; aucune valeur n'est affichée
    secrets = [
        settings.supabase_db_url.get_secret_value(),
        settings.openai_api_key.get_secret_value(),
    ]
    embeddings = CONFIG["embeddings"]
    embedder = CachedEmbedder(
        OpenAIEmbedder(
            model=embeddings["model"],
            dimensions=int(embeddings["dimensions"]),
            api_key=settings.openai_api_key.get_secret_value(),
            batch_size=int(embeddings["batch_size"]),
            max_attempts=int(embeddings["max_attempts"]),
        ),
        model=embeddings["model"],
        path=CACHE,
    )
    try:
        with psycopg.connect(settings.supabase_db_url.get_secret_value(), autocommit=True) as conn:
            conn.read_only = True  # garde-fou : aucune écriture possible dans cette session
            report = _evaluate(questions, embedder, conn)
    except Exception as error:  # réseau, SSL, base : on s'arrête et on montre l'erreur
        print("ÉCHEC (rien n'a été écrit) :\n  " + _error_chain(error, secrets))
        return 2

    print(f"Cache des questions : {embedder.hits} lus, {embedder.misses} calculés (OpenAI)")
    _print(report, questions)

    commit = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    today = dt.date.today().isoformat()
    output = RESULTS / f"retrieval_{args.set}_{today}_{commit}.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "set": args.set,
        "mode": args.mode,
        "date": today,
        "commit": commit,
        "working_tree_dirty": dirty,
        "config": {
            "embedding_model": embeddings["model"],
            "embedding_dimensions": int(embeddings["dimensions"]),
            "match_count": int(CONFIG["retrieval"]["match_count"]),
            "rrf_k": int(CONFIG["retrieval"]["rrf_k"]),
            "max_tokens": int(CONFIG["chunking"]["max_tokens"]),
            "min_quote_chars": int(CONFIG["guardrails"]["min_quote_chars"]),
            "configurations": {name: list(w) for name, w in _configs().items()},
        },
        "question_ids": [q["id"] for q in questions],
        **report,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"\nRésultats : {output.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
