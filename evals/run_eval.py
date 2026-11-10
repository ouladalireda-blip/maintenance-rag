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

Mode « grid » : grille de réglages FIXÉE À L'AVANCE, comparée à la sémantique seule.

    uv run python evals/run_eval.py --set dev --mode grid

- Poids (s, f, i) = (1, f, 2) avec f ∈ {0, 0.25, 0.5, 1}, match_count ∈ {10, 20} :
  8 configurations, aucune autre valeur. rrf_k de config.yaml.
- Rang d'une question = rang du premier bon passage parmi les match_count renvoyés (None
  sinon). R@1, R@3, R@10, R@20 (match_count 20 seulement) ; MRR tronqué au top 10
  (comparable entre match_count).
- Référence : sémantique seule (1, 0, 0) AU MÊME match_count. Comparaison appariée
  (victoires / défaites / égalités, absent = pire rang), ids qui sortent du top
  match_count alors qu'ils y étaient en sémantique seule, et ids qui y entrent.
- Règle de choix : a) rappel au match_count le plus haut de la grille (R@20 ; une
  configuration à match_count 10 ne renvoie rien au-delà du rang 10, son R@20 vaut donc son
  R@10) ; b) à égalité, f le plus petit puis match_count le plus petit ; c) MRR pour un ex
  æquo restant.

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
from metrics import (
    TOP_K,
    Candidate,
    first_hit_rank,
    hit,
    mrr,
    paired_comparison,
    recall_at_k,
    select_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
CACHE = ROOT / ".cache" / "query_embeddings.json"
RESULTS = ROOT / "evals" / "results"
DIAGNOSTIC_MATCH_COUNT = 10_000  # tous les candidats de chaque branche (69 passages en base)

# Grille fixée à l'avance : ne pas ajouter de valeur après avoir vu les résultats.
GRID_SEMANTIC_WEIGHT = 1.0
GRID_IDENTIFIER_WEIGHT = 2.0
GRID_FULL_TEXT_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
GRID_MATCH_COUNTS = (10, 20)
SEMANTIC_ONLY: Weights = (1.0, 0.0, 0.0)


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
        # Nombre de passages où la branche full-text matche : mesuré par hybrid_search elle-même
        # (poids 0,1,0, tous les candidats), jamais par une requête plein texte écrite ici.
        full_text_all = search(
            text, embedder, conn, configs["full-text seul"], DIAGNOSTIC_MATCH_COUNT, rrf_k=rrf_k
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
            "full_text_matches_total": sum(
                1 for r in full_text_all if r.full_text_rank is not None
            ),
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


def _grid() -> list[tuple[str, Weights, int]]:
    return [
        (f"f={f:g} mc={mc}", (GRID_SEMANTIC_WEIGHT, f, GRID_IDENTIFIER_WEIGHT), mc)
        for mc in GRID_MATCH_COUNTS
        for f in GRID_FULL_TEXT_WEIGHTS
    ]


def _ranks(
    questions: list[dict[str, Any]],
    embedder: Any,
    conn: Any,
    weights: Weights,
    match_count: int,
    rrf_k: int,
    min_chars: int,
) -> list[int | None]:
    ranks: list[int | None] = []
    for question in questions:
        results = search(question["question"], embedder, conn, weights, match_count, rrf_k=rrf_k)
        results = [result for result in results if result.rrf_score > 0]  # aucun remplissage
        ranks.append(first_hit_rank(results, question, min_chars, match_count))
    return ranks


def _metrics(ranks: list[int | None], match_count: int) -> dict[str, float | None]:
    top10 = [rank if rank is not None and rank <= TOP_K else None for rank in ranks]
    return {
        "recall@1": recall_at_k(ranks, 1),
        "recall@3": recall_at_k(ranks, 3),
        "recall@10": recall_at_k(ranks, 10),
        "recall@20": recall_at_k(ranks, 20) if match_count >= 20 else None,
        "recall_at_max_match_count": recall_at_k(ranks, max(GRID_MATCH_COUNTS)),
        "mrr@10": mrr(top10),
    }


def _evaluate_grid(questions: list[dict[str, Any]], embedder: Any, conn: Any) -> dict[str, Any]:
    rrf_k = int(CONFIG["retrieval"]["rrf_k"])
    min_chars = int(CONFIG["guardrails"]["min_quote_chars"])
    ids = [question["id"] for question in questions]

    baselines: dict[int, list[int | None]] = {
        mc: _ranks(questions, embedder, conn, SEMANTIC_ONLY, mc, rrf_k, min_chars)
        for mc in GRID_MATCH_COUNTS
    }
    reference = {
        f"sémantique seule mc={mc}": {
            "weights": list(SEMANTIC_ONLY),
            "match_count": mc,
            **_metrics(ranks, mc),
            "ranks": dict(zip(ids, ranks, strict=True)),
        }
        for mc, ranks in baselines.items()
    }

    grid: dict[str, dict[str, Any]] = {}
    for name, weights, mc in _grid():
        ranks = _ranks(questions, embedder, conn, weights, mc, rrf_k, min_chars)
        base = baselines[mc]
        paired = paired_comparison(ranks, base)
        grid[name] = {
            "weights": list(weights),
            "match_count": mc,
            **_metrics(ranks, mc),
            "vs_semantic_only": {
                "wins": paired.wins,
                "losses": paired.losses,
                "ties": paired.ties,
                "dropped_out": [
                    qid
                    for qid, rank, base_rank in zip(ids, ranks, base, strict=True)
                    if base_rank is not None and rank is None
                ],
                "entered": [
                    qid
                    for qid, rank, base_rank in zip(ids, ranks, base, strict=True)
                    if base_rank is None and rank is not None
                ],
            },
            "ranks": dict(zip(ids, ranks, strict=True)),
        }

    chosen = select_config(
        [
            Candidate(
                name,
                row["recall_at_max_match_count"],
                row["weights"][1],
                row["match_count"],
                row["mrr@10"],
            )
            for name, row in grid.items()
        ]
    )
    return {
        "reference": reference,
        "grid": grid,
        "selection_rule": (
            f"a) R@{max(GRID_MATCH_COUNTS)} le plus haut ; b) à égalité, f le plus petit puis "
            "match_count le plus petit ; c) MRR@10 pour un ex æquo restant"
        ),
        "selected": chosen.name,
    }


def _r(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _print_grid(report: dict[str, Any], questions: list[dict[str, Any]]) -> None:
    print("\nGrille (s=1, i=2), recherche seule avant reranking ; questions :", len(questions))
    print("Référence : sémantique seule (1,0,0) au même match_count ; absent = pire rang")
    print(
        f"{'configuration':24} {'poids':10} {'R@1':>5} {'R@3':>5} {'R@10':>5} {'R@20':>5} "
        f"{'MRR@10':>6} {'V/D/E':>8}  sortent du top | entrent dans le top"
    )
    for name, row in report["reference"].items():
        weights = ",".join(f"{w:g}" for w in row["weights"])
        print(
            f"{name:24} {weights:10} {_r(row['recall@1']):>5} {_r(row['recall@3']):>5} "
            f"{_r(row['recall@10']):>5} {_r(row['recall@20']):>5} {row['mrr@10']:6.3f} "
            f"{'réf.':>8}"
        )
    for name, row in report["grid"].items():
        weights = ",".join(f"{w:g}" for w in row["weights"])
        vs = row["vs_semantic_only"]
        record = f"{vs['wins']}/{vs['losses']}/{vs['ties']}"
        dropped = ", ".join(vs["dropped_out"]) or "aucun"
        entered = ", ".join(vs["entered"]) or "aucun"
        print(
            f"{name:24} {weights:10} {_r(row['recall@1']):>5} {_r(row['recall@3']):>5} "
            f"{_r(row['recall@10']):>5} {_r(row['recall@20']):>5} {row['mrr@10']:6.3f} "
            f"{record:>8}  {dropped} | {entered}"
        )
    print(f"\nRègle : {report['selection_rule']}")
    chosen = report["grid"][report["selected"]]
    weights = ",".join(f"{w:g}" for w in chosen["weights"])
    print(
        f"Configuration retenue : {report['selected']} (poids {weights}, match_count "
        f"{chosen['match_count']}, R@{max(GRID_MATCH_COUNTS)} "
        f"{chosen['recall_at_max_match_count']:.2f})"
    )


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

    print("\nc) Passages où la branche full-text de hybrid_search a matché (poids 0,1,0, tous")
    print("   candidats) ; ident : branche identifiants (appel diagnostic hybride)")
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
    parser.add_argument("--mode", choices=["retrieval", "grid"], required=True)
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
            if args.mode == "grid":
                report = _evaluate_grid(questions, embedder, conn)
            else:
                report = _evaluate(questions, embedder, conn)
    except Exception as error:  # réseau, SSL, base : on s'arrête et on montre l'erreur
        print("ÉCHEC (rien n'a été écrit) :\n  " + _error_chain(error, secrets))
        return 2

    print(f"Cache des questions : {embedder.hits} lus, {embedder.misses} calculés (OpenAI)")
    if args.mode == "grid":
        _print_grid(report, questions)
    else:
        _print(report, questions)

    commit = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    today = dt.date.today().isoformat()
    prefix = "grid" if args.mode == "grid" else "retrieval"
    output = RESULTS / f"{prefix}_{args.set}_{today}_{commit}.json"
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
            "rrf_k": int(CONFIG["retrieval"]["rrf_k"]),
            "max_tokens": int(CONFIG["chunking"]["max_tokens"]),
            "min_quote_chars": int(CONFIG["guardrails"]["min_quote_chars"]),
            **(
                {
                    "grid": {
                        name: {"weights": list(w), "match_count": mc} for name, w, mc in _grid()
                    },
                    "reference_weights": list(SEMANTIC_ONLY),
                }
                if args.mode == "grid"
                else {
                    "match_count": int(CONFIG["retrieval"]["match_count"]),
                    "configurations": {name: list(w) for name, w in _configs().items()},
                }
            ),
        },
        "question_ids": [q["id"] for q in questions],
        **report,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"\nRésultats : {output.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
