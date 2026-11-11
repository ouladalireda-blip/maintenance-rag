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

Mode « rerank » : FlashRank sur les candidats de la recherche retenue (poids, match_count et
rrf_k de config.yaml), sur TOUTES les questions (valides et pièges). τ n'est PAS choisi ici.

    uv run python evals/run_eval.py --set dev --mode rerank

- a) Questions valides : R@1, R@3, MRR avant / après reranking (rangs parmi les match_count
  candidats), comparaison appariée après contre avant, questions dont le rang bouge de plus
  de 2 places.
- b) Chaque question : score maximal du reranker, score de la bonne fiche (valides : score du
  premier bon passage après reranking), catégorie ; trié par score maximal décroissant.
- c) Séparation : min / médiane / max du score maximal par groupe (valides, chaque catégorie
  de piège) ; pièges dont le score maximal dépasse le plus petit score maximal d'une valide.
- d) Latence du seul reranking par question (modèle chargé, un appel d'échauffement non
  compté) : médiane et p95.

Le reranker est retiré du pipeline (docs/adr/0002-retrait-reranker-flashrank.md,
rerank.enabled: false) ; le mode rerank reste disponible pour tester un autre modèle : il charge
le modèle pour l'évaluation seulement, en le signalant.

Mode « signals » : signaux de la recherche retenue (poids, match_count et rrf_k de config.yaml),
sur TOUTES les questions. Aucun seuil n'est choisi ici.

    uv run python evals/run_eval.py --set dev --mode signals

- a) Questions valides : R@1, R@3, R@5, R@8, R@10 (combien de passages donner au LLM).
- b) Chaque question : similarité cosinus du meilleur candidat (le 1er de la recherche
  retenue), écart avec le 2e (sim 1er − sim 2e, négatif si le 2e est plus proche), rang
  sémantique du 1er, catégorie ; trié par similarité décroissante.
- c) Séparation de cette similarité par groupe (valides, chaque catégorie de piège) et pièges
  au-dessus de la plus petite similarité d'une valide.
- Similarité = 1 − (embedding <=> vecteur de la question), l'opérateur de la branche
  sémantique de hybrid_search, recalculé exactement (sans l'index HNSW) par une requête en
  lecture seule sur les ids renvoyés : hybrid_search ne renvoie que des rangs.

Mode « e2e » : pipeline complet (recherche retenue → top_k passages → LLM → garde-fou 2) sur
TOUTES les questions, comparé à des critères FIXÉS À L'AVANCE (evals/metrics.py).

    uv run python evals/run_eval.py --set dev --mode e2e --estimate-only   # coût seulement
    uv run python evals/run_eval.py --set dev --mode e2e
    uv run python evals/run_eval.py --set dev --mode e2e --runs 3          # stabilité

- Jeu test : refusé sauf ``--set test --mode e2e --runs 3 --final`` (protocole de l'ADR 0003 :
  3 runs comptés comme une évaluation, verdict sur la moyenne, réponses dangereuses par run,
  scores aussi sur le sous-ensemble des questions dont le passage cible n'est visé par aucune
  question dev). Ce sous-ensemble est calculé AVANT tout appel au LLM.
- --runs N : N runs successifs, fichiers suffixés _run1…_runN (jamais écrasés : arrêt si l'un
  existe), coût estimé pour les N runs avant tout appel, puis résumé de stabilité (statuts qui
  changent, min / max et OK par run de chaque critère).

- Arrêt avant tout appel si llm.model ou un prix est vide dans config.yaml.
- Estimation du coût AVANT l'éval : tokens d'entrée (prompt système + schéma + question +
  passages, comptés en cl100k_base, +10 % de marge) et plafond de sortie (max_output_tokens
  par question), aux prix de config.yaml. Arrêt si l'estimation dépasse 1 $.
- Toute erreur (réseau, SSL, API) arrête l'éval : elle n'est jamais comptée comme un refus.
- Pièges : CRR = refusés / pièges ; réponses dangereuses = pièges ANSWERED.
- Valides : FRR = refusées / valides ; bonne source = ANSWERED dont une affirmation cite un
  passage qui contient la citation de référence (``hit``).
- Réponses complètes (texte du LLM, extraits du guide) : evals/runs/ (non versionné).
  evals/results/ : ids, statuts, codes de raison et nombres seulement.

Base en LECTURE SEULE (connexion read_only : hybrid_search, et en mode signals une lecture des
distances sur public.chunks). Les vecteurs des
questions sont mis en cache dans .cache/query_embeddings.json (non versionné).
"""

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, LiteralString

import yaml

from maintenance_rag.db import vector_literal
from maintenance_rag.rerank import Reranker, rerank
from maintenance_rag.retrieval import (
    CachedEmbedder,
    SearchResult,
    Weights,
    load_retrieval_settings,
    search,
)
from metrics import (
    DANGEROUS_RATE,
    MAX_DANGEROUS,
    MAX_DANGEROUS_RATE,
    MAX_FRR,
    MIN_GOOD_SOURCE,
    TOP_K,
    Candidate,
    criteria_stability,
    dangerous_rate_criterion,
    e2e_criteria,
    first_hit_rank,
    hit,
    mrr,
    paired_comparison,
    percentile,
    recall_at_k,
    select_config,
    separation,
    spread,
    status_stability,
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
    return {
        "hybride": load_retrieval_settings().weights,
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


MOVE_THRESHOLD = 2  # « le rang change de plus de 2 places »


def _evaluate_rerank(
    questions: list[dict[str, Any]], embedder: Any, conn: Any, reranker: Reranker
) -> dict[str, Any]:
    settings = load_retrieval_settings()
    min_chars = int(CONFIG["guardrails"]["min_quote_chars"])
    worst = settings.match_count + 1  # rang d'un bon passage absent (pour mesurer un déplacement)

    retrieved = [(question, search(question["question"], embedder, conn)) for question in questions]
    if retrieved and retrieved[0][1]:  # échauffement du modèle : non compté dans la latence
        rerank(retrieved[0][0]["question"], retrieved[0][1], reranker)

    per_question: dict[str, dict[str, Any]] = {}
    for question, results in retrieved:
        start = time.perf_counter()
        reranked = rerank(question["question"], results, reranker)
        latency_ms = (time.perf_counter() - start) * 1000
        row: dict[str, Any] = {
            "category": question["category"],
            "must_refuse": bool(question["must_refuse"]),
            "candidates": len(results),
            "max_score": reranked[0].score if reranked else None,
            "latency_ms": latency_ms,
            "reranked": [
                {"chunk_id": r.result.chunk_id, "search_rank": r.search_rank, "score": r.score}
                for r in reranked
            ],
        }
        if not question["must_refuse"]:
            before = first_hit_rank(results, question, min_chars, settings.match_count)
            after = first_hit_rank(
                [r.result for r in reranked], question, min_chars, settings.match_count
            )
            row.update(
                rank_before=before,
                rank_after=after,
                gold_score=reranked[after - 1].score if after else None,
            )
        per_question[question["id"]] = row

    valid = [qid for qid, row in per_question.items() if not row["must_refuse"]]
    before_ranks = [per_question[qid]["rank_before"] for qid in valid]
    after_ranks = [per_question[qid]["rank_after"] for qid in valid]
    paired = paired_comparison(after_ranks, before_ranks)

    def quality(ranks: list[int | None]) -> dict[str, float]:
        return {
            "recall@1": recall_at_k(ranks, 1),
            "recall@3": recall_at_k(ranks, 3),
            f"recall@{settings.match_count}": recall_at_k(ranks, settings.match_count),
            "mrr": mrr(ranks),
        }

    moved = []
    for qid in valid:
        row = per_question[qid]
        before, after = row["rank_before"] or worst, row["rank_after"] or worst
        if abs(after - before) > MOVE_THRESHOLD:
            moved.append({"id": qid, "rank_before": row["rank_before"], "rank_after": after})

    groups: dict[str, list[str]] = {"valid": valid}
    for qid, row in per_question.items():
        if row["must_refuse"]:
            groups.setdefault(row["category"], []).append(qid)
    max_scores = {qid: row["max_score"] for qid, row in per_question.items()}
    separation = {
        name: {"count": len(ids), **spread([max_scores[qid] for qid in ids])}
        for name, ids in groups.items()
    }
    lowest_valid = separation["valid"]["min"]
    above = {
        name: [qid for qid in ids if max_scores[qid] > lowest_valid]
        for name, ids in groups.items()
        if name != "valid"
    }
    latencies = [row["latency_ms"] for row in per_question.values()]
    return {
        "quality": {
            "before": quality(before_ranks),
            "after": quality(after_ranks),
            "after_vs_before": paired._asdict(),
            "moved_more_than_2": moved,
        },
        "separation": {
            "max_score_by_group": separation,
            "lowest_valid_max_score": lowest_valid,
            "traps_above_lowest_valid": above,
            "traps_above_lowest_valid_count": sum(len(ids) for ids in above.values()),
        },
        "latency_ms": {
            "median": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "questions": len(latencies),
        },
        "per_question": per_question,
    }


def _s(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _print_rerank(report: dict[str, Any]) -> None:
    quality = report["quality"]
    keys = list(quality["before"])
    valid_count = sum(1 for row in report["per_question"].values() if not row["must_refuse"])
    print(f"\na) Questions valides : {valid_count} ; rangs parmi les candidats de la recherche")
    print(f"{'':8} " + " ".join(f"{key:>10}" for key in keys))
    for label in ("before", "after"):
        name = "avant" if label == "before" else "après"
        print(f"{name:8} " + " ".join(f"{quality[label][key]:>10.3f}" for key in keys))
    vs = quality["after_vs_before"]
    print(f"Après contre avant (V/D/E, absent = pire) : {vs['wins']}/{vs['losses']}/{vs['ties']}")
    print(
        f"Rang qui bouge de plus de {MOVE_THRESHOLD} places : {len(quality['moved_more_than_2'])}"
    )
    for row in quality["moved_more_than_2"]:
        print(f"  {row['id']:14} {_fmt(row['rank_before']):>3} -> {_fmt(row['rank_after']):>3}")

    print("\nb) Chaque question, triée par score maximal décroissant")
    print(
        f"{'id':14} {'catégorie':16} {'score max':>9} {'bonne fiche':>11} {'rang av.':>8} "
        f"{'rang ap.':>8}"
    )
    rows = sorted(
        report["per_question"].items(), key=lambda item: (-(item[1]["max_score"] or 0), item[0])
    )
    for qid, row in rows:
        print(
            f"{qid:14} {row['category']:16} {_s(row['max_score']):>9} "
            f"{_s(row.get('gold_score')):>11} {_fmt(row.get('rank_before')):>8} "
            f"{_fmt(row.get('rank_after')):>8}"
        )

    separation = report["separation"]
    print("\nc) Séparation du score maximal (τ non choisi)")
    print(f"{'groupe':16} {'n':>3} {'min':>8} {'médiane':>8} {'max':>8}")
    for name, row in separation["max_score_by_group"].items():
        print(
            f"{name:16} {row['count']:>3} {row['min']:8.4f} {row['median']:8.4f} {row['max']:8.4f}"
        )
    print(
        f"Plus petit score maximal d'une valide : {separation['lowest_valid_max_score']:.4f} ; "
        f"pièges au-dessus : {separation['traps_above_lowest_valid_count']}"
    )
    for name, ids in separation["traps_above_lowest_valid"].items():
        print(f"  {name:16} {len(ids):>2}  {', '.join(ids) or '-'}")

    latency = report["latency_ms"]
    print(
        f"\nd) Latence du reranking ({latency['questions']} questions) : "
        f"médiane {latency['median']:.0f} ms, p95 {latency['p95']:.0f} ms"
    )


SIGNAL_RECALL_KS = (1, 3, 5, 8, 10)
COSINE_SIMILARITY: LiteralString = """
    select c.id, 1 - (c.embedding <=> %s::extensions.vector)
    from public.chunks c
    where c.id = any(%s)
"""


def _similarities(conn: Any, vector: list[float], chunk_ids: list[int]) -> dict[int, float]:
    with conn.cursor() as cursor:
        cursor.execute(COSINE_SIMILARITY, (vector_literal(vector), chunk_ids))
        return {int(chunk_id): float(similarity) for chunk_id, similarity in cursor.fetchall()}


def _evaluate_signals(questions: list[dict[str, Any]], embedder: Any, conn: Any) -> dict[str, Any]:
    settings = load_retrieval_settings()
    min_chars = int(CONFIG["guardrails"]["min_quote_chars"])

    per_question: dict[str, dict[str, Any]] = {}
    for question in questions:
        results = search(question["question"], embedder, conn)
        vector = embedder.embed([question["question"]])[0]
        similarity = _similarities(conn, vector, [r.chunk_id for r in results])
        ordered = [similarity[r.chunk_id] for r in results]
        row: dict[str, Any] = {
            "category": question["category"],
            "must_refuse": bool(question["must_refuse"]),
            "candidates": len(results),
            "top1_chunk_id": results[0].chunk_id if results else None,
            "top1_semantic_rank": results[0].semantic_rank if results else None,
            "top1_similarity": ordered[0] if ordered else None,
            "gap_1_2": ordered[0] - ordered[1] if len(ordered) > 1 else None,
            "max_similarity": max(ordered, default=None),
            "similarities": ordered,  # dans l'ordre de la recherche retenue
        }
        if not question["must_refuse"]:
            row["rank"] = first_hit_rank(results, question, min_chars, settings.match_count)
        per_question[question["id"]] = row

    valid = [qid for qid, row in per_question.items() if not row["must_refuse"]]
    ranks = [per_question[qid]["rank"] for qid in valid]
    groups: dict[str, list[str]] = {"valid": valid}
    for qid, row in per_question.items():
        if row["must_refuse"]:
            groups.setdefault(row["category"], []).append(qid)
    top1 = {qid: row["top1_similarity"] for qid, row in per_question.items()}
    return {
        "recall": {f"recall@{k}": recall_at_k(ranks, k) for k in SIGNAL_RECALL_KS},
        "separation_top1_similarity": separation(top1, groups),
        "top1_is_semantic_rank1": sum(
            1 for row in per_question.values() if row["top1_semantic_rank"] == 1
        ),
        "per_question": per_question,
    }


def _print_signals(report: dict[str, Any]) -> None:
    rows = report["per_question"]
    valid_count = sum(1 for row in rows.values() if not row["must_refuse"])
    print(f"\na) Questions valides : {valid_count} ; rang du bon passage dans la recherche retenue")
    print("   " + "  ".join(f"{key}={value:.3f}" for key, value in report["recall"].items()))

    print("\nb) Chaque question, triée par similarité cosinus du 1er candidat décroissante")
    print(
        f"   1er candidat = rang sémantique 1 pour {report['top1_is_semantic_rank1']}/{len(rows)}"
    )
    print(f"{'id':14} {'catégorie':16} {'sim 1er':>8} {'écart 1-2':>9} {'sem':>4} {'rang':>4}")
    ordered = sorted(rows.items(), key=lambda item: (-(item[1]["top1_similarity"] or 0), item[0]))
    for qid, row in ordered:
        print(
            f"{qid:14} {row['category']:16} {_s(row['top1_similarity']):>8} "
            f"{_s(row['gap_1_2']):>9} {_fmt(row['top1_semantic_rank']):>4} "
            f"{_fmt(row.get('rank')):>4}"
        )

    sep = report["separation_top1_similarity"]
    print("\nc) Séparation de la similarité du 1er candidat (aucun seuil choisi)")
    print(f"{'groupe':16} {'n':>3} {'min':>8} {'médiane':>8} {'max':>8}")
    for name, row in sep["by_group"].items():
        print(
            f"{name:16} {row['count']:>3} {row['min']:8.4f} {row['median']:8.4f} {row['max']:8.4f}"
        )
    above = sep["above_lowest_reference"]
    print(
        f"Plus petite similarité d'une valide : {sep['lowest_reference']:.4f} ; pièges au-dessus : "
        f"{sum(len(ids) for ids in above.values())}"
    )
    for name, ids in above.items():
        total = sep["by_group"][name]["count"]
        print(f"  {name:16} {len(ids):>2}/{total:<2} {', '.join(ids) or '-'}")


BUDGET_USD = 1.0  # arrêt si l'estimation du coût de l'éval complète dépasse ce montant
TOKEN_MARGIN = 1.10  # cl100k_base ≠ tokenizer exact du modèle : marge sur l'entrée
RUNS = ROOT / "evals" / "runs"  # réponses complètes : non versionné (.gitignore)


def _estimate(
    questions: list[dict[str, Any]], embedder: Any, conn: Any, settings: Any
) -> dict[str, Any]:
    """Coût maximal estimé de l'éval complète, sans appeler le LLM."""
    from maintenance_rag.chunking import count_tokens
    from maintenance_rag.generation import (
        SCHEMA,
        SYSTEM_PROMPT,
        build_user_message,
        label_passages,
    )

    fixed = count_tokens(SYSTEM_PROMPT) + count_tokens(json.dumps(SCHEMA, ensure_ascii=False))
    counted = 0
    for question in questions:
        passages = search(question["question"], embedder, conn)[: settings.top_k]
        message = build_user_message(question["question"], label_passages(passages))
        counted += fixed + count_tokens(message)
    input_tokens = math.ceil(counted * TOKEN_MARGIN)
    output_tokens = settings.llm.max_output_tokens * len(questions)
    cost = settings.llm.cost(input_tokens, output_tokens)
    return {
        "questions": len(questions),
        "input_tokens_cl100k": counted,
        "input_tokens_with_margin": input_tokens,
        "output_tokens_max": output_tokens,
        "price_per_1m_input": settings.llm.price_per_1m_input,
        "price_per_1m_output": settings.llm.price_per_1m_output,
        "cost_usd_max": cost,
        "budget_usd": BUDGET_USD,
    }


def _print_estimate(estimate: dict[str, Any], model: str) -> None:
    print(f"\nEstimation du coût ({model}, {estimate['questions']} questions)")
    print(
        f"  entrée : {estimate['input_tokens_cl100k']} tokens cl100k_base, "
        f"{estimate['input_tokens_with_margin']} avec +10 % "
        f"× {estimate['price_per_1m_input']} $/M"
    )
    print(
        f"  sortie : {estimate['output_tokens_max']} tokens au plus (max_output_tokens × questions)"
        f" × {estimate['price_per_1m_output']} $/M"
    )
    print(
        f"  coût maximal estimé : {estimate['cost_usd_max']:.4f} $ "
        f"(budget {estimate['budget_usd']:.2f} $)"
    )


# Critère ajouté avant la mesure v2 : le CRR ne doit pas baisser sous son niveau v1 (20/21).
MIN_CRR = (20, 21)


CHUNK_PAGES: LiteralString = """
    select c.id, c.content,
           case when c.page_start - d.page_offset > 0 then c.page_start - d.page_offset end,
           case when c.page_end - d.page_offset > 0 then c.page_end - d.page_offset end
    from public.chunks c
    join public.documents d on d.id = c.document_id
"""
FINAL_RUNS = 3  # ADR 0003 : 3 runs lancés ensemble, comptés comme UNE évaluation


def meets_threshold(name: str, value: float | None) -> bool:
    """Le seuil fixé à l'avance du critère ``name`` est-il respecté par ``value`` ?"""
    if value is None:
        return False
    thresholds = {
        "réponses dangereuses": value <= MAX_DANGEROUS,
        DANGEROUS_RATE: value <= MAX_DANGEROUS_RATE,
        "FRR": value <= MAX_FRR,
        "bonne source": value >= MIN_GOOD_SOURCE,
        "CRR": value * MIN_CRR[1] >= MIN_CRR[0],
    }
    if name not in thresholds:
        raise ValueError(f"critère inconnu : {name!r}")
    return thresholds[name]


# ADR 0003 : critères jugés sur le PIRE run plutôt que sur la moyenne (valeur la plus haute).
WORST_RUN_CRITERIA = frozenset({DANGEROUS_RATE})


def mean_verdict(per_run: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    """ADR 0003 : par critère, moyenne / min / max sur les runs, puis verdict.

    Le verdict porte sur la MOYENNE, sauf pour les critères de ``WORST_RUN_CRITERIA`` (taux de
    réponses dangereuses du jeu test), jugés sur le PIRE run. Une valeur non calculable (None)
    est exclue de la moyenne ; sans aucune valeur, ou si le pire run n'est pas calculable, le
    critère n'est pas atteint.
    """
    summary = []
    for row in criteria_stability(per_run):
        known = [value for value in row["values"] if value is not None]
        mean = sum(known) / len(known) if known else None
        worst_run = row["name"] in WORST_RUN_CRITERIA
        if worst_run:
            aggregate = None if None in row["values"] else max(row["values"])
        else:
            aggregate = mean
        summary.append(
            {
                **row,
                "mean": mean,
                "aggregation": "pire run" if worst_run else "moyenne",
                "aggregate": aggregate,
                "met": meets_threshold(row["name"], aggregate),
            }
        )
    return summary


def criteria_for(
    set_name: str,
    dangerous: int,
    traps: int,
    valid_refused: int,
    valid: int,
    good_source: int,
    valid_answered: int,
) -> list[dict[str, Any]]:
    """Critères d'un run. Jeu dev : ceux fixés avant la première mesure, inchangés. Jeu test
    (ADR 0003) : les réponses dangereuses deviennent un taux (≤ 5 % des pièges)."""
    rows = e2e_criteria(dangerous, valid_refused, valid, good_source, valid_answered) + [
        _crr_criterion(traps - dangerous, traps)
    ]
    if set_name == "test":
        rows = [
            dangerous_rate_criterion(dangerous, traps)
            if row["name"] == "réponses dangereuses"
            else row
            for row in rows
        ]
    return rows


def target_chunks(
    questions: Sequence[dict[str, Any]],
    chunks: Sequence[tuple[int, str, int | None, int | None]],
    min_chars: int,
) -> dict[str, set[int]]:
    """Passages cibles de chaque question valide : chunks (id, contenu, page imprimée de début,
    de fin) qui contiennent sa citation de référence et couvrent sa page attendue."""
    from maintenance_rag.normalize import quote_matches

    targets: dict[str, set[int]] = {}
    for question in questions:
        if question["must_refuse"]:
            continue
        page = question["expected_page"]
        targets[question["id"]] = {
            chunk_id
            for chunk_id, content, start, end in chunks
            if start is not None
            and end is not None
            and start <= page <= end
            and quote_matches(question["exact_quote"], content, min_chars)
        }
    return targets


def unseen_in_dev(
    questions: Sequence[dict[str, Any]],
    targets: dict[str, set[int]],
    dev_targets: dict[str, set[int]],
) -> tuple[list[str], list[str]]:
    """Sous-ensemble « passages jamais vus en dev » (ADR 0003), et valides écartées faute de
    passage cible connu.

    Une question valide en fait partie si aucun de ses passages cibles n'est visé par une
    question dev. Les pièges n'ont pas de passage cible : ils y sont tous (CRR et réponses
    dangereuses restent ainsi calculables sur le sous-ensemble).
    """
    seen: set[int] = set().union(*dev_targets.values()) if dev_targets else set()
    subset: list[str] = []
    unknown: list[str] = []
    for question in questions:
        qid = question["id"]
        if question["must_refuse"]:
            subset.append(qid)
        elif not targets.get(qid):
            unknown.append(qid)
        elif not targets[qid] & seen:
            subset.append(qid)
    return subset, unknown


def _crr_criterion(refused: int, traps: int) -> dict[str, Any]:
    low, high = MIN_CRR
    return {
        "name": "CRR",
        "value": refused / traps,
        "threshold": f"≥ {low}/{high}",
        "ok": refused * high >= low * traps,
    }


def _e2e_metrics(
    questions: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    verdicts: dict[str, Any],
    set_name: str = "dev",
) -> dict[str, Any]:
    from maintenance_rag.guardrails import refused_reasons

    traps = [q["id"] for q in questions if q["must_refuse"]]
    valid = [q["id"] for q in questions if not q["must_refuse"]]
    answered = "ANSWERED"
    dangerous = [qid for qid in traps if records[qid]["status"] == answered]
    valid_answered = [qid for qid in valid if records[qid]["status"] == answered]
    valid_refused = [qid for qid in valid if records[qid]["status"] != answered]
    good = [qid for qid in valid_answered if records[qid]["good_source"]]
    not_good = [qid for qid in valid_answered if not records[qid]["good_source"]]

    def by_llm(ids: list[str]) -> list[str]:
        return [qid for qid in ids if records[qid]["status"] == "REFUSED_BY_LLM"]

    def by_guardrail(ids: list[str]) -> dict[str, int]:
        return refused_reasons([verdicts[qid] for qid in ids])

    costs = [records[qid]["cost_usd"] or 0.0 for qid in records]
    totals = [records[qid]["latency_ms"]["total"] for qid in records]
    steps = sorted({step for r in records.values() for step in r["latency_ms"]})
    return {
        "traps": {
            "count": len(traps),
            "refused": len(traps) - len(dangerous),
            "crr": (len(traps) - len(dangerous)) / len(traps),
            "dangerous_answers": dangerous,
        },
        "valid": {
            "count": len(valid),
            "refused": valid_refused,
            "frr": len(valid_refused) / len(valid),
            "answered": len(valid_answered),
            "good_source": len(good),
            "good_source_rate": len(good) / len(valid_answered) if valid_answered else None,
            "answered_without_expected_source": not_good,
            "expected_passage_not_given": [
                qid for qid in valid if not records[qid]["gold_in_passages"]
            ],
        },
        "refusals": {
            "by_llm": {"traps": by_llm(traps), "valid": by_llm(valid)},
            "by_guardrail": {"traps": by_guardrail(traps), "valid": by_guardrail(valid)},
        },
        "cost_usd": {"total": sum(costs), "per_question": sum(costs) / len(costs)},
        "tokens": {
            "input": sum(r["input_tokens"] for r in records.values()),
            "output": sum(r["output_tokens"] for r in records.values()),
        },
        "latency_ms": {
            "median": percentile(totals, 50),
            "p95": percentile(totals, 95),
            "median_by_step": {
                step: percentile(
                    [r["latency_ms"][step] for r in records.values() if step in r["latency_ms"]],
                    50,
                )
                for step in steps
            },
        },
        "criteria": criteria_for(
            set_name,
            len(dangerous),
            len(traps),
            len(valid_refused),
            len(valid),
            len(good),
            len(valid_answered),
        ),
    }


def _print_e2e(metrics: dict[str, Any], records: dict[str, dict[str, Any]]) -> None:
    print("\nPar question")
    print(
        f"{'id':14} {'catégorie':16} {'statut':21} {'source':>6} {'attendu∈P':>9} "
        f"{'coût $':>8} {'ms':>6}  raison"
    )
    for qid, row in records.items():
        source = "-" if row["good_source"] is None else ("oui" if row["good_source"] else "NON")
        given = (
            "-"
            if row["gold_in_passages"] is None
            else ("oui" if row["gold_in_passages"] else "non")
        )
        cost = "-" if row["cost_usd"] is None else f"{row['cost_usd']:.5f}"
        print(
            f"{qid:14} {row['category']:16} {row['status']:21} {source:>6} {given:>9} "
            f"{cost:>8} {row['latency_ms']['total']:6.0f}  {row['reason']}"
        )

    traps, valid = metrics["traps"], metrics["valid"]
    print(
        f"\nPièges ({traps['count']}) : CRR = {traps['refused']}/{traps['count']} "
        f"= {traps['crr']:.2f}"
    )
    print(
        f"  réponses dangereuses (ANSWERED) : {len(traps['dangerous_answers'])} "
        f"{', '.join(traps['dangerous_answers']) or ''}"
    )
    print(
        f"Valides ({valid['count']}) : FRR = {len(valid['refused'])}/{valid['count']} = "
        f"{valid['frr']:.2f}  {', '.join(valid['refused'])}"
    )
    rate = valid["good_source_rate"]
    print(
        f"  bonne source : {valid['good_source']}/{valid['answered']} ANSWERED"
        f" = {'-' if rate is None else f'{rate:.2f}'}"
    )
    print(
        "  ANSWERED sans citer le passage attendu (à relire) : "
        f"{', '.join(valid['answered_without_expected_source']) or 'aucune'}"
    )
    print(
        "  passage attendu absent des passages fournis : "
        f"{', '.join(valid['expected_passage_not_given']) or 'aucune'}"
    )

    refusals = metrics["refusals"]
    print("\nRépartition des refus")
    for group in ("traps", "valid"):
        name = "pièges" if group == "traps" else "valides"
        llm_ids = refusals["by_llm"][group]
        reasons = refusals["by_guardrail"][group]
        print(
            f"  {name:8} par le LLM : {len(llm_ids)} ; par le garde-fou 2 : {sum(reasons.values())}"
        )
        for reason, count in reasons.items():
            print(f"           {reason} : {count}")

    cost, tokens, latency = metrics["cost_usd"], metrics["tokens"], metrics["latency_ms"]
    print(
        f"\nCoût : {cost['total']:.4f} $ au total, {cost['per_question']:.5f} $ par question "
        f"({tokens['input']} tokens d'entrée, {tokens['output']} de sortie)"
    )
    steps = ", ".join(f"{k} {v:.0f}" for k, v in latency["median_by_step"].items())
    print(f"Latence : médiane {latency['median']:.0f} ms, p95 {latency['p95']:.0f} ms ({steps})")

    print("\nCritères fixés à l'avance")
    for row in metrics["criteria"]:
        value = row["value"]
        shown = "-" if value is None else (f"{value:.2f}" if isinstance(value, float) else value)
        print(
            f"  {row['name']:22} {shown!s:>6} {row['threshold']:>6}  "
            f"{'OK' if row['ok'] else 'ÉCHEC'}"
        )


def _one_run(
    pipeline: Any,
    seen: dict[int, SearchResult],
    questions: list[dict[str, Any]],
    runs_path: Path,
    min_chars: int,
    label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Un passage du pipeline sur toutes les questions ; réponses complètes dans ``runs_path``."""
    records: dict[str, dict[str, Any]] = {}
    verdicts: dict[str, Any] = {}
    RUNS.mkdir(parents=True, exist_ok=True)
    with runs_path.open("x", encoding="utf-8") as runs:  # « x » : jamais d'écrasement
        for number, question in enumerate(questions, start=1):
            result = pipeline.answer(question["question"])
            given = [seen[chunk_id] for chunk_id in result.passage_ids]
            valid = not question["must_refuse"]
            cited = [c.chunk_id for c in result.citations]
            record = {
                "category": question["category"],
                "must_refuse": bool(question["must_refuse"]),
                "status": result.status.value,
                "reason_code": result.verdict.reason.value if result.verdict.reason else None,
                "reason": result.reason,
                "passage_ids": result.passage_ids,
                "cited_chunk_ids": cited,
                "gold_in_passages": any(hit(p, question, min_chars) for p in given)
                if valid
                else None,
                "good_source": any(hit(seen[chunk_id], question, min_chars) for chunk_id in cited)
                if valid and result.status.value == "ANSWERED"
                else None,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "latency_ms": result.latencies_ms,
            }
            records[question["id"]] = record
            verdicts[question["id"]] = result.verdict
            runs.write(
                json.dumps(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        **record,
                        "model": result.model,
                        "prompt_version": result.prompt_version,
                        "reponse": result.reponse,
                        "llm_output": result.llm_output.model_dump() if result.llm_output else None,
                        "citations": [
                            {
                                "passage_id": c.passage_id,
                                "chunk_id": c.chunk_id,
                                "section_title": c.section_title,
                                "printed_page_start": c.printed_page_start,
                                "printed_page_end": c.printed_page_end,
                                "texte": c.texte,
                                "extrait_exact": c.extrait_exact,
                            }
                            for c in result.citations
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            runs.flush()
            print(f"  {label}{number:>2}/{len(questions)} {question['id']} {result.status.value}")
    return records, verdicts


def _print_run_summary(label: str, metrics: dict[str, Any]) -> None:
    traps, valid = metrics["traps"], metrics["valid"]
    rate = valid["good_source_rate"]
    print(f"\n{label}")
    print(
        f"  réponses dangereuses : {len(traps['dangerous_answers'])}/{traps['count']} "
        f"= {len(traps['dangerous_answers']) / traps['count']:.3f} "
        f"{', '.join(traps['dangerous_answers']) or ''}"
    )
    print(f"  CRR : {traps['refused']}/{traps['count']} = {traps['crr']:.3f}")
    print(
        f"  FRR : {len(valid['refused'])}/{valid['count']} = {valid['frr']:.3f}  "
        f"{', '.join(valid['refused'])}"
    )
    print(
        f"  bonne source : {valid['good_source']}/{valid['answered']} = "
        f"{'-' if rate is None else f'{rate:.3f}'}"
        f" ; ANSWERED sans le passage attendu : "
        f"{', '.join(valid['answered_without_expected_source']) or 'aucune'}"
    )
    print(f"  coût : {metrics['cost_usd']['total']:.4f} $")


def _print_stability(
    labels: list[str],
    questions: list[dict[str, Any]],
    all_records: list[dict[str, dict[str, Any]]],
    all_metrics: list[dict[str, Any]],
) -> None:
    stability = status_stability(
        [{qid: row["status"] for qid, row in records.items()} for records in all_records]
    )
    category = {q["id"]: q["category"] for q in questions}
    print(
        f"\nb) Questions dont le statut diffère d'un run à l'autre : {len(stability['differing'])}"
    )
    if stability["differing"]:
        print(f"{'id':14} {'catégorie':16} " + " ".join(f"{label:21}" for label in labels))
        for qid, statuses in stability["differing"].items():
            print(f"{qid:14} {category[qid]:16} " + " ".join(f"{s:21}" for s in statuses))
            reasons = [all_records[i][qid]["reason"] or "-" for i in range(len(labels))]
            print(f"{'':31} raisons : {' | '.join(reasons)}")

    print("\nc) Critères fixés à l'avance, sur chaque run")
    print(
        f"{'critère':22} {'min':>6} {'max':>6} {'seuil':>8}  "
        + " ".join(f"{label:>6}" for label in labels)
    )
    for row in criteria_stability([metrics["criteria"] for metrics in all_metrics]):

        def shown(value: Any) -> str:
            if value is None:
                return "-"
            return f"{value:.3f}" if isinstance(value, float) else str(value)

        print(
            f"{row['name']:22} {shown(row['min']):>6} {shown(row['max']):>6} "
            f"{row['threshold']:>8}  "
            + " ".join(f"{'OK' if ok else 'ÉCHEC':>6}" for ok in row["ok"])
        )
    every = all(row["all_ok"] for row in criteria_stability([m["criteria"] for m in all_metrics]))
    print(f"  les 4 critères OK sur les {len(labels)} runs : {'oui' if every else 'NON'}")
    print(
        f"\nd) Questions au statut identique sur les {len(labels)} runs : "
        f"{stability['identical']}/{stability['total']}"
    )


def _run_e2e(
    args: argparse.Namespace,
    questions: list[dict[str, Any]],
    env: Any,
    embedder: Any,
    secrets: Sequence[str],
) -> int:
    import psycopg

    from maintenance_rag.llm import OpenAILLM
    from maintenance_rag.pipeline import Pipeline, PipelineSettings

    settings = PipelineSettings.from_config()
    llm_settings = settings.llm
    if not llm_settings.model:
        print("ARRÊT : llm.model est vide dans config/config.yaml. Renseigne un petit modèle")
        print("OpenAI actuel qui supporte les Structured Outputs ; aucun appel n'a été fait.")
        return 3
    if llm_settings.price_per_1m_input is None or llm_settings.price_per_1m_output is None:
        print("ARRÊT : llm.price_per_1m_input / price_per_1m_output vides dans config.yaml ;")
        print("le coût ne peut pas être estimé. Aucun appel n'a été fait.")
        return 3

    min_chars = int(CONFIG["guardrails"]["min_quote_chars"])
    commit = _git("rev-parse", "--short", "HEAD")
    today = dt.date.today().isoformat()
    count = args.runs or 1
    suffixes = [f"_run{i}" for i in range(1, count + 1)] if args.runs else [""]
    labels = [s.lstrip("_") or "run" for s in suffixes]
    stem = f"e2e_{args.set}_{today}_{commit}"
    targets = [(RESULTS / f"{stem}{s}.json", RUNS / f"{stem}{s}.jsonl") for s in suffixes]
    summary_path = RESULTS / f"{stem}_summary.json"
    if args.runs:
        existing = [p for pair in targets for p in pair if p.exists()]
        existing += [summary_path] if summary_path.exists() else []
        if existing:
            print(
                "ARRÊT : ces fichiers existent déjà, rien n'est écrasé ; aucun appel n'a été fait :"
            )
            for path in existing:
                print(f"  {path.relative_to(ROOT).as_posix()}")
            return 6

    all_records: list[dict[str, dict[str, Any]]] = []
    all_verdicts: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    try:
        with psycopg.connect(env.supabase_db_url.get_secret_value(), autocommit=True) as conn:
            conn.read_only = True  # garde-fou : aucune écriture possible dans cette session
            subset: list[str] | None = None
            if args.set == "test":
                with conn.cursor() as cursor:
                    cursor.execute(CHUNK_PAGES)
                    chunks = [(int(r[0]), str(r[1]), r[2], r[3]) for r in cursor.fetchall()]
                dev_questions = yaml.safe_load(
                    (ROOT / "evals" / "questions_dev.yaml").read_text("utf-8")
                )["questions"]
                subset, unknown = unseen_in_dev(
                    questions,
                    target_chunks(questions, chunks, min_chars),
                    target_chunks(dev_questions, chunks, min_chars),
                )
                valid_in = sum(1 for q in questions if q["id"] in subset and not q["must_refuse"])
                print(
                    f"Sous-ensemble « passages jamais vus en dev » : {len(subset)} questions "
                    f"({valid_in} valides + tous les pièges) ; valides sans passage cible : "
                    f"{', '.join(unknown) or 'aucune'}"
                )
            estimate = _estimate(questions, embedder, conn, settings)
            _print_estimate(estimate, llm_settings.model)
            total_max = estimate["cost_usd_max"] * count
            if count > 1:
                print(
                    f"  {count} runs : coût maximal estimé {total_max:.4f} $ "
                    f"(budget {BUDGET_USD:.2f} $)"
                )
            if total_max > BUDGET_USD:
                print(f"ARRÊT : estimation supérieure au budget de {BUDGET_USD:.2f} $.")
                return 4
            if args.estimate_only:
                return 0

            reranker = None
            if settings.rerank_enabled:
                from maintenance_rag.rerank import FlashRankReranker

                reranker = FlashRankReranker.from_config()
            seen: dict[int, SearchResult] = {}

            def retrieve(question: str) -> list[SearchResult]:
                results = search(question, embedder, conn)
                seen.update({result.chunk_id: result for result in results})
                return results

            llm = OpenAILLM(llm_settings, api_key=env.openai_api_key.get_secret_value())
            pipeline = Pipeline(retrieve, llm, settings, reranker)
            for index, (label, (output, runs_path)) in enumerate(zip(labels, targets, strict=True)):
                prefix = f"{label} " if args.runs else ""
                records, verdicts = _one_run(
                    pipeline, seen, questions, runs_path, min_chars, prefix
                )
                metrics = _e2e_metrics(questions, records, verdicts, args.set)
                if not args.runs:
                    _print_e2e(metrics, records)
                payload = {
                    "set": args.set,
                    "mode": args.mode,
                    "date": today,
                    "commit": commit,
                    "run": index + 1 if args.runs else None,
                    "runs_total": count if args.runs else None,
                    "working_tree_dirty": bool(_git("status", "--porcelain")),
                    "config": {
                        "llm_model": llm_settings.model,
                        "temperature": llm_settings.temperature,
                        "max_output_tokens": llm_settings.max_output_tokens,
                        "prompt_version": _prompt_version(),
                        "top_k": settings.top_k,
                        "min_quote_chars": settings.min_quote_chars,
                        "rerank_enabled": settings.rerank_enabled,
                        "retrieval_weights": list(load_retrieval_settings().weights),
                        "match_count": load_retrieval_settings().match_count,
                    },
                    "estimate": estimate,
                    **metrics,
                    "per_question": records,
                }
                RESULTS.mkdir(parents=True, exist_ok=True)
                with (
                    output.open("x", encoding="utf-8")
                    if args.runs
                    else output.open("w", encoding="utf-8") as handle
                ):
                    handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                print(f"  métriques : {output.relative_to(ROOT).as_posix()}")
                print(
                    "  réponses complètes (non versionnées) : "
                    f"{runs_path.relative_to(ROOT).as_posix()}"
                )
                all_records.append(records)
                all_verdicts.append(verdicts)
                all_metrics.append(metrics)
    except Exception as error:  # réseau, SSL, API, base : arrêt, jamais compté comme un refus
        print(
            f"ÉCHEC après {len(all_records)} run(s) complet(s) ; le run en cours n'a pas de "
            "métriques :\n  " + _error_chain(error, secrets)
        )
        return 2

    if args.runs:
        print("\na) Résumé par run")
        for label, metrics in zip(labels, all_metrics, strict=True):
            _print_run_summary(label, metrics)
        _print_stability(labels, questions, all_records, all_metrics)
        verdict = mean_verdict([metrics["criteria"] for metrics in all_metrics])
        _print_mean_verdict(
            "e) Verdict (ADR 0003 : moyenne des runs, pire run pour le taux)", verdict
        )
        summary: dict[str, Any] = {
            "set": args.set,
            "date": today,
            "commit": commit,
            "runs": [pair[0].relative_to(ROOT).as_posix() for pair in targets],
            "dangerous_answers_by_run": [
                {
                    "count": len(m["traps"]["dangerous_answers"]),
                    "traps": m["traps"]["count"],
                    "rate": len(m["traps"]["dangerous_answers"]) / m["traps"]["count"],
                    "ids": m["traps"]["dangerous_answers"],
                }
                for m in all_metrics
            ],
            "verdict_on_mean": verdict,
            "stability": status_stability(
                [{qid: row["status"] for qid, row in r.items()} for r in all_records]
            ),
        }
        if subset is not None:
            in_subset = [q for q in questions if q["id"] in set(subset)]
            subset_metrics = [
                _e2e_metrics(
                    in_subset,
                    {qid: r[qid] for qid in subset},
                    {qid: v[qid] for qid in subset},
                    args.set,
                )
                for r, v in zip(all_records, all_verdicts, strict=True)
            ]
            subset_verdict = mean_verdict([m["criteria"] for m in subset_metrics])
            _print_mean_verdict(
                f"f) Sous-ensemble « passages jamais vus en dev » ({len(subset)} questions)",
                subset_verdict,
            )
            summary["unseen_in_dev"] = {"question_ids": subset, "verdict_on_mean": subset_verdict}
        with summary_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(f"\nSynthèse : {summary_path.relative_to(ROOT).as_posix()}")
    return 0


def _print_mean_verdict(title: str, verdict: list[dict[str, Any]]) -> None:
    def shown(value: Any) -> str:
        if value is None:
            return "-"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    print(f"\n{title}")
    print(
        f"{'critère':28} {'moyenne':>8} {'min':>6} {'max':>6} {'seuil':>8} {'jugé sur':>9} "
        f"{'valeur':>7}  verdict"
    )
    for row in verdict:
        print(
            f"{row['name']:28} {shown(row['mean']):>8} {shown(row['min']):>6} "
            f"{shown(row['max']):>6} {row['threshold']:>8} {row['aggregation']:>9} "
            f"{shown(row['aggregate']):>7}  {'ATTEINT' if row['met'] else 'NON ATTEINT'}"
        )


def _prompt_version() -> str:
    from maintenance_rag.generation import PROMPT_VERSION

    return PROMPT_VERSION


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
    parser.add_argument("--set", choices=["dev", "test"], required=True)
    parser.add_argument(
        "--mode", choices=["retrieval", "grid", "rerank", "signals", "e2e"], required=True
    )
    parser.add_argument(
        "--estimate-only", action="store_true", help="e2e : estimer le coût sans appeler le LLM"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="e2e, jeu dev : N runs successifs (fichiers suffixés _run1…_runN, jamais écrasés)",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="seul moyen d'évaluer le jeu test : mesure finale, une seule fois",
    )
    args = parser.parse_args(argv)
    if args.set == "test":
        print("AVERTISSEMENT : le jeu test ne doit être utilisé qu'une seule fois.")
        if not args.final:
            print("REFUS : --set test exige le drapeau explicite --final ; rien n'a été lancé.")
            return 5
    if args.set == "test" and (args.mode != "e2e" or args.runs != FINAL_RUNS):
        print(
            f"REFUS : l'éval finale (ADR 0003) se lance en une fois : --mode e2e --runs "
            f"{FINAL_RUNS} --final ; rien n'a été lancé."
        )
        return 5
    if args.runs is not None and (args.mode != "e2e" or args.runs < 1):
        print("REFUS : --runs N (N ≥ 1) n'existe qu'en mode e2e.")
        return 5

    data = yaml.safe_load((ROOT / "evals" / f"questions_{args.set}.yaml").read_text("utf-8"))
    all_questions = args.mode in {"rerank", "signals", "e2e"}
    questions = [q for q in data["questions"] if all_questions or not q["must_refuse"]]

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
    if args.mode == "e2e":
        return _run_e2e(args, questions, settings, embedder, secrets)
    try:
        reranker = None
        if args.mode == "rerank":  # seul endroit où le modèle est chargé
            from maintenance_rag.rerank import FlashRankReranker

            if not CONFIG["rerank"]["enabled"]:
                print("rerank.enabled = false : modèle chargé pour cette évaluation seulement")
            reranker = FlashRankReranker.from_config()  # cache local uniquement
        with psycopg.connect(settings.supabase_db_url.get_secret_value(), autocommit=True) as conn:
            conn.read_only = True  # garde-fou : aucune écriture possible dans cette session
            if args.mode == "grid":
                report = _evaluate_grid(questions, embedder, conn)
            elif reranker is not None:
                report = _evaluate_rerank(questions, embedder, conn, reranker)
            elif args.mode == "signals":
                report = _evaluate_signals(questions, embedder, conn)
            else:
                report = _evaluate(questions, embedder, conn)
    except Exception as error:  # réseau, SSL, base : on s'arrête et on montre l'erreur
        print("ÉCHEC (rien n'a été écrit) :\n  " + _error_chain(error, secrets))
        return 2

    print(f"Cache des questions : {embedder.hits} lus, {embedder.misses} calculés (OpenAI)")
    if args.mode == "grid":
        _print_grid(report, questions)
    elif args.mode == "rerank":
        _print_rerank(report)
    elif args.mode == "signals":
        _print_signals(report)
    else:
        _print(report, questions)

    commit = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    today = dt.date.today().isoformat()
    output = RESULTS / f"{args.mode}_{args.set}_{today}_{commit}.json"
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
                    "match_count": load_retrieval_settings().match_count,
                    "weights": list(load_retrieval_settings().weights),
                    "rerank_model": CONFIG["rerank"]["model"],
                    "rerank_max_length": int(CONFIG["rerank"]["max_length"]),
                    "rerank_text": "section_title + \\n + content",
                }
                if args.mode == "rerank"
                else {
                    "match_count": load_retrieval_settings().match_count,
                    "weights": list(load_retrieval_settings().weights),
                    "similarity": "1 - (embedding <=> question), exacte, sur les ids renvoyés",
                    "best_candidate": "1er de la recherche retenue",
                }
                if args.mode == "signals"
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
