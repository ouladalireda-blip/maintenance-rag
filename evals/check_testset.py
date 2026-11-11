"""Contrôle d'INDÉPENDANCE entre le jeu dev et le jeu test, sans exécuter le système.

    uv run python evals/check_testset.py

Ni recherche, ni pipeline, ni LLM : le jeu test reste jamais vu par le système. Le script lit :

- evals/questions_dev.yaml et evals/questions_test.yaml ;
- les passages en base (lecture seule, ``public.chunks``), pour savoir quel passage contient
  chaque citation de référence ;
- les embeddings des QUESTIONS (pas des passages), mis en cache dans
  .cache/testset_question_embeddings.json, séparé du cache de l'évaluation.

Proximités signalées, test ↔ dev puis à l'intérieur du jeu test :

a) même ``exact_quote`` normalisée, ou citations qui se chevauchent de plus de 50 % (plus longue
   sous-chaîne commune des formes normalisées, rapportée à la plus courte des deux) ;
b) même passage cible (un même chunk contient les deux citations, à la bonne page) : à
   signaler, pas forcément à retirer ;
c) questions trop proches : similarité cosinus des embeddings de questions > 0,90.

Sortie console uniquement : ids et nombres, jamais de texte du guide. Rien n'est écrit, hormis
le cache des embeddings de questions.
"""

import collections
import difflib
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
CACHE = ROOT / ".cache" / "testset_question_embeddings.json"
OVERLAP_THRESHOLD = 0.50
COSINE_THRESHOLD = 0.90

CHUNKS_SQL = """
    select c.id, c.content,
           case when c.page_start - d.page_offset > 0 then c.page_start - d.page_offset end,
           case when c.page_end - d.page_offset > 0 then c.page_end - d.page_offset end
    from public.chunks c
    join public.documents d on d.id = c.document_id
"""


def _load(name: str) -> list[dict[str, Any]]:
    data = yaml.safe_load((ROOT / "evals" / name).read_text(encoding="utf-8"))
    return list(data["questions"])


def quote_overlap(a: str, b: str) -> float:
    """Plus longue sous-chaîne commune des deux formes normalisées / longueur de la plus courte."""
    if not a or not b:
        return 0.0
    match = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match()
    return match.size / min(len(a), len(b))


def cosine(u: list[float], v: list[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v, strict=True))
    return dot / (math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v)))


def chapter(question: dict[str, Any]) -> str:
    section = question.get("expected_section")
    if not section:
        return "(piège)"
    head = section.split("›")[0].strip()
    number = re.match(r"(\d+)", head)
    return number.group(1) if number else head


def gold_chunks(
    question: dict[str, Any], chunks: list[tuple[int, str, int | None, int | None]]
) -> set[int]:
    from maintenance_rag.normalize import quote_matches

    page = question["expected_page"]
    return {
        chunk_id
        for chunk_id, content, start, end in chunks
        if start is not None
        and end is not None
        and start <= page <= end
        and quote_matches(
            question["exact_quote"], content, int(CONFIG["guardrails"]["min_quote_chars"])
        )
    }


def proximities(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    quotes: dict[str, str],
    targets: dict[str, set[int]],
    vectors: dict[str, list[float]],
    same_set: bool,
) -> tuple[list[tuple[str, str, str, str]], dict[str, tuple[str, float]]]:
    """Lignes (id gauche, id droite, type, valeur) et, pour chaque question de gauche, la plus
    proche de droite au sens du cosinus."""
    rows: list[tuple[str, str, str, str]] = []
    nearest: dict[str, tuple[str, float]] = {}
    for i, a in enumerate(left):
        others = right[i + 1 :] if same_set else right
        best = ("-", -1.0)
        for b in others:
            ida, idb = a["id"], b["id"]
            if ida in quotes and idb in quotes:
                qa, qb = quotes[ida], quotes[idb]
                if qa == qb:
                    rows.append((ida, idb, "a) même citation", "1.00"))
                else:
                    overlap = quote_overlap(qa, qb)
                    if overlap > OVERLAP_THRESHOLD:
                        rows.append((ida, idb, "a) citations qui se chevauchent", f"{overlap:.2f}"))
                shared = targets[ida] & targets[idb]
                if shared:
                    rows.append((ida, idb, "b) même passage cible", f"chunk {sorted(shared)}"))
            similarity = cosine(vectors[ida], vectors[idb])
            if similarity > COSINE_THRESHOLD:
                rows.append((ida, idb, "c) questions proches (cosinus)", f"{similarity:.3f}"))
            if similarity > best[1]:
                best = (idb, similarity)
        nearest[a["id"]] = best
    return rows, nearest


def _table(title: str, rows: list[tuple[str, str, str, str]], right_label: str) -> None:
    print(f"\n{title} : {len(rows)} signalement(s)")
    if not rows:
        return
    print(f"{'id test':15} {right_label:15} {'type de proximité':34} valeur")
    for row in rows:
        print(f"{row[0]:15} {row[1]:15} {row[2]:34} {row[3]}")


def _distribution(questions: list[dict[str, Any]]) -> None:
    for key, getter in (
        ("catégorie", lambda q: q["category"]),
        ("style", lambda q: q.get("style", "-")),
        ("difficulté", lambda q: q.get("difficulty", "-")),
        ("chapitre", chapter),
    ):
        counts = collections.Counter(getter(q) for q in questions)
        print(f"  {key:10} : " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    by_category = collections.defaultdict(collections.Counter)
    for q in questions:
        by_category[q["category"]][chapter(q)] += 1
    print(
        "  chapitre par catégorie (valides) : "
        + ", ".join(f"ch. {k} : {v}" for k, v in sorted(by_category["valid"].items()))
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    import psycopg

    from maintenance_rag.embeddings import OpenAIEmbedder
    from maintenance_rag.ingestion import Settings
    from maintenance_rag.normalize import normalize_for_match
    from maintenance_rag.retrieval import CachedEmbedder

    dev, test = _load("questions_dev.yaml"), _load("questions_test.yaml")
    settings = Settings()  # lit .env ; aucune valeur n'est affichée
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
    with psycopg.connect(settings.supabase_db_url.get_secret_value(), autocommit=True) as conn:
        conn.read_only = True  # aucune écriture possible dans cette session
        with conn.cursor() as cursor:
            cursor.execute(CHUNKS_SQL)
            chunks = [(int(r[0]), str(r[1]), r[2], r[3]) for r in cursor.fetchall()]

    everything = dev + test
    quotes = {
        q["id"]: normalize_for_match(q["exact_quote"]) for q in everything if not q["must_refuse"]
    }
    targets = {q["id"]: gold_chunks(q, chunks) for q in everything if not q["must_refuse"]}
    texts = [q["question"] for q in everything]
    vectors = dict(zip((q["id"] for q in everything), embedder.embed(texts), strict=True))
    print(
        f"Questions : dev {len(dev)}, test {len(test)} ; passages en base : {len(chunks)} ; "
        f"embeddings de questions : {embedder.hits} en cache, {embedder.misses} calculés"
    )
    no_target = [qid for qid, found in targets.items() if not found]
    print(f"Citations sans passage cible en base : {', '.join(no_target) or 'aucune'}")

    rows, nearest = proximities(test, dev, quotes, targets, vectors, same_set=False)
    _table("Test ↔ dev", rows, "id dev")
    inner, _ = proximities(test, test, quotes, targets, vectors, same_set=True)
    _table("Doublons à l'intérieur du jeu test", inner, "autre id test")

    closest = sorted(nearest.items(), key=lambda item: -item[1][1])
    median = sorted(value for _, value in nearest.values())[len(nearest) // 2]
    print(
        "\nCosinus question test ↔ question dev la plus proche : "
        f"max {closest[0][1][1]:.3f}, médiane {median:.3f}"
    )
    print("  10 plus proches : " + ", ".join(f"{t}~{d} {s:.3f}" for t, (d, s) in closest[:10]))

    print("\nRépartition du jeu test")
    _distribution(test)
    print("Répartition du jeu dev (rappel)")
    _distribution(dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
