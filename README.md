# maintenance-rag — Assistant consignation sur le guide INRS ED 6109

Assistant pour techniciens de maintenance qui répond **uniquement** à partir du guide INRS ED 6109
(*Consignations et déconsignations*), cite la page source mot pour mot, et **refuse** quand la réponse
précise n'est pas dans le guide.

En sécurité industrielle, une procédure inventée est plus dangereuse qu'un « je ne sais pas » : le projet
mesure donc d'abord **le taux de réponses dangereuses** et **le taux de refus corrects**, pas seulement la
pertinence des réponses.

> ⚠️ Outil d'aide à la lecture du guide. La procédure de consignation du site et le chargé de consignation
> font foi.

<!-- Captures : docs/img/reponse.png, docs/img/refus-utile.png, docs/img/refus.png -->

---

## Résultats

Deux jeux d'évaluation écrits **avant** le code : un jeu **dev** (50 questions, sert à régler) et un jeu
**test** (115 questions, figé dans git, jamais utilisé pour régler, évalué **une seule fois**, 3 runs).
Protocole : [ADR 0003](docs/adr/0003-protocole-eval-finale.md). Détails : [docs/evaluation.md](docs/evaluation.md).

| Critère (fixé avant la mesure) | Seuil | Dev (50 q.) | **Test (115 q., 3 runs)** | Verdict test |
|---|---|---|---|---|
| Pièges refusés (CRR) | ≥ 95 % | 95 % | **95,5 %** (min 94,2) | ✅ |
| Bonne source quand il répond | ≥ 80 % | 96 % | **89,7 %** | ✅ |
| Réponses dangereuses (pire run) | ≤ 5 % des pièges | 1/21 | **2 à 3 sur 52** (5,8 % au pire) | ❌ de justesse |
| Faux refus (FRR) | ≤ 20 % | 17 % | **33 %** | ❌ |
| Stabilité (statut identique sur 3 runs) | — | 47/50 | **111/115** | |
| Coût par question | — | 0,0005 $ | **0,0005 $** | |
| Latence médiane | — | 2,1 s | ≈ 2 s | |

**Lecture honnête.** Le système est **sûr** (les 12 pièges à référence normative sont tous refusés, et
quand il répond il cite la bonne page dans 90 % des cas) mais **trop prudent** : il refuse une vraie question
sur trois. L'écart dev → test sur le FRR (17 % → 33 %) mesure le surapprentissage du prompt v2 réglé sur le
dev. La cause est identifiée : dans 16 refus sur 21, le bon passage était fourni au LLM, qui a jugé
l'information « non explicite » parce que la question était formulée loin du vocabulaire du guide (FRR de 22 %
sur les questions faciles, 43 % sur les difficiles). Aucun réglage n'a été fait après avoir vu ces chiffres.

Une relecture manuelle de la **fidélité des résumés** complète ces chiffres (le garde-fou vérifie qu'une
citation existe, pas qu'elle prouve l'affirmation) : sur le dev, 21 résumés fidèles, 2 incomplets, 1 trompeur
(4,2 %).

---

## Comment ça marche

```
INGESTION (une fois)
PDF ─► lecteur structuré (PyMuPDF) ─► blocs typés ─► 69 passages ≤ 350 tokens
    ─► identifiants (NF C18-510, R. 4411-6, R1…) + embeddings 1536 ─► Supabase / pgvector

QUESTION
question ─► recherche (sémantique + références exactes, fusion RRF) ─► 8 passages
         ─► LLM, sortie structurée : { info_presente, réponse, affirmations[texte, extrait_exact, passage] }
         ─► garde-fou déterministe : chaque extrait existe mot pour mot, chaque nombre est cité
         ─► réponse avec citations et pages   OU   refus
```

**Ce qui décide du refus.** Pas un score de similarité : un passage proche du sujet n'implique pas que la
réponse précise y soit (un *near miss* parle du bon sujet sans donner la valeur demandée). Le refus repose
sur le **LLM** (`info_presente`) puis sur une **vérification par le code** : une seule citation introuvable
ou un seul chiffre non cité, et toute la réponse est refusée. Dans l'interface, **la citation du guide est
la réponse principale** ; le résumé du LLM vient en second, marqué « à vérifier dans l'extrait ».

**Stack.** Python 3.12 · uv · PyMuPDF · OpenAI (`text-embedding-3-small`, `gpt-4o-mini`, température 0,
Structured Outputs) · Supabase (PostgreSQL + pgvector, HNSW) · FastAPI · Streamlit · Docker · pytest ·
ruff · mypy strict · GitHub Actions.

---

## Décisions prises sur mesure (pas sur réputation)

| Décision | Ce que la mesure a montré | Doc |
|---|---|---|
| **Taille des passages : 350 tokens** | Mesurée avec le tokenizer du reranker prévu (fenêtre 512) : à 500, 10 passages débordaient ; à 350, 63 tokens de marge pour la question | `docs/evaluation.md` |
| **Full-text désactivé par configuration** | Sans IDF, `ts_rank_cd` est bruité sur un corpus homogène (« consign » dans 56 passages sur 69) : il faisait sortir 3 bons passages du top 10 sur 29 questions. Grille de réglages fixée à l'avance → poids 0, 20 candidats, R@20 = 29/29 | migrations 0003–0004 |
| **Reranker retiré** | FlashRank MultiBERT sature en français : R@1 de 0,55 → 0,03, une recette de cuisine notée 0,93 pour une question de consignation | [ADR 0002](docs/adr/0002-retrait-reranker-flashrank.md) |
| **Refus confié au LLM + vérification déterministe** | Ni le reranker ni la similarité ne séparent les pièges (18 pièges sur 21 au-dessus de la pire question valide) | [ADR 0001](docs/adr/0001-double-garde-fou.md) |
| **3 runs, moyenne et pire cas** | À température 0, le FRR variait de 14 à 24 % d'un run à l'autre : un seul run ne suffit pas | [ADR 0003](docs/adr/0003-protocole-eval-finale.md) |

Le prompt v1 donnait 28 % de faux refus sur le dev. Le diagnostic question par question a montré que la
moitié venait d'artefacts de mise en forme (puces, appels de note) que le LLM retirait à raison ; les
corrections ont été choisies selon leur **risque** (un assouplissement qui aurait laissé passer les pièges a
été refusé). v2 : 17 %, sans qu'aucun piège ne change de statut.

---

## Jeu d'évaluation

- **4 catégories** : `valid` (réponse dans le guide, citation exacte et page attendues), `near_miss` (sujet
  présent, valeur absente), `identifier_trap` (référence citée dans le guide, information absente),
  `out_of_scope` (réponse seulement dans une image, ou hors sujet).
- **Dev** : 50 questions (29 valides, 21 pièges). **Test** : 115 questions (63 valides, 52 pièges), styles
  variés (atelier, SMS sans accents, contexte, formel), 47 difficiles.
- Rédigées avec l'aide d'un LLM **différent** de celui du système, puis **vérifiées une à une** : chaque
  citation est contrôlée mot pour mot dans le PDF, à la bonne page et dans un seul passage (tests
  automatiques) ; chaque piège est vérifié à la main (Ctrl+F sur les termes clés).
- **Indépendance dev/test** contrôlée (`evals/check_testset.py`) : aucune question test à plus de 0,70 de
  similarité d'une question dev, au plus 3 questions par passage, résultats aussi publiés sur les 82
  questions dont le passage n'est visé par aucune question dev (FRR 37,8 %, CRR identique).
- Le jeu test est **verrouillé** : `run_eval.py --set test` refuse sans `--final`, et impose 3 runs.

---

## Limites connues et v3 envisagée

| Limite | Piste |
|---|---|
| **Une citation exacte ne prouve pas l'affirmation** : 3 pièges du test (052, 058, 027) et quelques résumés sont passés avec une citation vraie mais hors propos | Vérificateur d'implication (second appel structuré : « cet extrait prouve-t-il cette affirmation ? ») |
| **FRR 33 %** : le LLM refuse quand la question s'éloigne du vocabulaire du guide, même avec le bon passage | Assouplir la règle « information explicite » **avec** le vérificateur ci-dessus, régler sur le dev, évaluer sur un **nouveau** jeu test |
| Recherche : 5 refus sur 21 sans le bon passage dans les 8 | `top_k` 10–12 |
| Reranker | Tester un cross-encodeur multilingue récent avec le mode `rerank` existant |
| Un seul document, pas de gestion des versions ni des droits | Hors périmètre du prototype |

Règle du projet : **aucune modification du système après lecture des résultats du test** ; toute v3 sera
réglée sur le dev et évaluée sur un jeu test neuf.

---

## Lancer

### En local

```sh
uv sync                                   # Python 3.12
cp .env.example .env                      # OPENAI_API_KEY, SUPABASE_DB_URL (jamais versionnés)
# ingestion (une fois, après avoir appliqué supabase/migrations/0001 à 0004)
uv run python -m maintenance_rag.ingestion --doc inrs-ed6109            # dry-run : compte et coûts
uv run python -m maintenance_rag.ingestion --doc inrs-ed6109 --apply    # ~70 passages, < 0,01 $
# API + interface
uv run uvicorn maintenance_rag.api.main:app --port 8000
uv run streamlit run ui/app.py            # http://localhost:8501
```

Le PDF INRS n'est **pas versionné** (© INRS, reproduction interdite) : il est lu localement à l'ingestion ;
seul le texte indexé vit en base privée (RLS, droits limités à `service_role`). Voir
[docs/corpus.md](docs/corpus.md).

### Avec Docker

```sh
docker compose up -d --build
docker compose ps                         # api et ui « healthy »
```

Interface : http://localhost:8600 · API : http://localhost:8100/docs. Le `.env` est lu au lancement, jamais
copié dans les images ; les images ne contiennent ni le PDF, ni le jeu test. Sans `.env`, l'API démarre en
mode dégradé (`/health` 200, `/ready` 503) — c'est ce que vérifie la CI, **sans réseau**.

**Derrière un antivirus ou un proxy qui inspecte HTTPS** (`UnknownIssuer` au build) : exporter le certificat
racine public de l'inspecteur dans `certs/` (jamais versionné) ; les images l'ajoutent à leur magasin système,
sans désactiver aucune vérification. Procédure détaillée dans la section Docker de l'historique du dépôt et
dans `Dockerfile.api`.

### Évaluer

```sh
uv run python evals/run_eval.py --set dev --mode retrieval      # Recall@k, MRR, par branche de recherche
uv run python evals/run_eval.py --set dev --mode e2e --runs 3   # CRR, FRR, bonne source, réponses dangereuses
uv run python evals/fidelity.py review && uv run python evals/fidelity.py summary   # relecture manuelle
```

---

## Qualité

230 tests (`pytest`), typage strict (`mypy`), lint et format (`ruff`), CI GitHub Actions qui construit aussi
les images Docker et vérifie l'absence de secrets. Chaque réponse enregistrée en évaluation est une
structure d'ids et de nombres : aucun texte du guide n'est versionné.

```sh
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest
```

---

## Structure

```
src/maintenance_rag/
  pdf_reader.py      lecture structurée du PDF (colonnes, titres, encadrés, notes)
  chunking.py        découpage structurel (≤ 350 tokens, listes jamais coupées)
  identifiers.py     normes, articles, règlements, procédés R1–R4 → forme canonique
  ingestion.py       dry-run, sha256, transaction unique, idempotent
  retrieval.py       hybrid_search (RRF), réglages depuis config.yaml
  generation.py      prompt versionné, sortie structurée (JSON strict)
  guardrails.py      vérification déterministe des citations et des nombres
  pipeline.py        orchestration, statuts ANSWERED / REFUSED_BY_LLM / REFUSED_BY_GUARDRAIL
  api/               FastAPI : /health, /ready, /ask, /feedback, budget journalier, journal SQLite
ui/                  Streamlit, client de l'API (citation en premier)
evals/               jeux dev et test, métriques, run_eval, check_testset, fidelity, résultats
supabase/migrations/ 0001 schéma · 0002 has_warning · 0003 full-text français · 0004 ordre des dictionnaires
docs/adr/            0001 double garde-fou · 0002 retrait du reranker · 0003 protocole d'éval finale
```

---

## Méthode

Code écrit avec l'aide de Claude Code ; conception de l'évaluation, critères, diagnostics et décisions
relus et arbitrés à la main. Les décisions structurantes sont consignées dans des ADR, et les chiffres
publiés sont ceux du jeu test figé, pas ceux du jeu de réglage.

*Projet personnel — Reda Oulad Ali, Master AMDI (Université Gustave Eiffel).*
