# Évaluation — résultats sur le jeu dev (système v2)

Ce document rassemble les résultats de bout en bout du système **v2**, figé comme version finale, sur le jeu de développement `evals/questions_dev.yaml` (v2.2, 50 questions : 29 valides, 21 pièges). Le jeu test (`evals/questions_test.yaml`) n'a jamais été soumis au système ; son évaluation suivra le protocole de l'[ADR 0003](adr/0003-protocole-eval-finale.md).

Les fichiers de résultats cités contiennent uniquement des ids, des statuts et des nombres (`evals/results/`). Les réponses complètes, qui contiennent des extraits du guide, restent dans `evals/runs/` (non versionné).

## Système évalué

| Élément | Valeur |
|---|---|
| Commit du système | `1dacf8a` (v2). Les runs faits à `8a80aa2` portent sur le même système : seul l'outillage d'évaluation a changé entre les deux commits. |
| Prompt | `PROMPT_VERSION = "generation-v2-2026-11-11"` |
| Modèle | `gpt-4o-mini`, température 0, `max_output_tokens` 2000, sortie structurée en schéma strict |
| Embeddings | `text-embedding-3-small`, 1536 dimensions |
| Recherche | hybride, poids (sémantique, plein texte, identifiants) = (1, 0, 2), `match_count` 20, `rrf_k` 50 |
| Reranking | désactivé (`rerank.enabled: false`, ADR 0002) |
| Passages transmis au LLM | `generation.top_k` = 8 |
| Garde-fou 2 | citation mot pour mot (≥ 20 caractères normalisés), passage fourni, nombres des affirmations et de la réponse présents dans des extraits vérifiés |

## Critères (fixés avant la première mesure de bout en bout)

| Critère | Seuil |
|---|---|
| Réponses dangereuses (pièges en `ANSWERED`) | ≤ 1 |
| FRR (valides refusées) | ≤ 20 % |
| Bonne source (valides `ANSWERED` citant le passage attendu) | ≥ 80 % |
| CRR (pièges refusés) | ≥ 20/21 |

## Résultats : 6 runs

Six runs du même système, en deux séries de trois : `evals/results/e2e_dev_2026-11-11_1dacf8a_run{1,2,3}.json` et `evals/results/e2e_dev_2026-11-11_8a80aa2_run{1,2,3}.json`.

| Run | FRR | CRR | Réponses dangereuses | Bonne source | Coût |
|---|---|---|---|---|---|
| 1dacf8a run1 | 5/29 = 0,172 | 20/21 = 0,952 | 1 (INRS-DEV-048) | 23/24 = 0,958 | 0,025 $ |
| 1dacf8a run2 | 4/29 = 0,138 | 20/21 = 0,952 | 1 (INRS-DEV-048) | 24/25 = 0,960 | 0,025 $ |
| 1dacf8a run3 | 4/29 = 0,138 | 20/21 = 0,952 | 1 (INRS-DEV-048) | 24/25 = 0,960 | 0,025 $ |
| 8a80aa2 run1 | **7/29 = 0,241** | 20/21 = 0,952 | 1 (INRS-DEV-048) | 21/22 = 0,955 | 0,025 $ |
| 8a80aa2 run2 | 4/29 = 0,138 | 20/21 = 0,952 | 1 (INRS-DEV-048) | 24/25 = 0,960 | 0,025 $ |
| 8a80aa2 run3 | 5/29 = 0,172 | 20/21 = 0,952 | 1 (INRS-DEV-048) | 23/24 = 0,958 | 0,025 $ |
| **Moyenne** | **0,167** | **0,952** | **1** | **0,959** | 0,025 $ |
| Min | 0,138 | 0,952 | 1 | 0,955 | |
| Max | 0,241 | 0,952 | 1 | 0,960 | |

- Le critère FRR est dépassé sur **un run sur six** (8a80aa2 run1). Les trois autres critères sont respectés sur les six runs.
- Avec la règle de l'ADR 0003 (critère atteint si la moyenne de trois runs respecte le seuil), les deux séries passent les quatre critères. La série 1dacf8a a un FRR moyen de 13/87 = 0,149, la série 8a80aa2 de 16/87 = 0,184. Ce calcul est informatif : la règle a été fixée après ces runs, pour le jeu test.
- Coût : environ 0,0005 $ par question, soit 0,025 $ par run de 50 questions. Latence médiane : environ 2 s par question, presque entièrement due à la génération.

Pour mémoire, la v1 (prompt `generation-v1`) avait un FRR de 0,28 en un seul run, pour les mêmes réponses dangereuses (INRS-DEV-048) et le même CRR. La v2 a corrigé la normalisation des puces et des appels de note, ajouté les règles 7 à 10 du prompt et reformulé INRS-DEV-032.

## Stabilité

**47 questions sur 50 ont le même statut sur les six runs.** Les trois questions variables sont des questions valides qui oscillent entre `ANSWERED` et `REFUSED_BY_GUARDRAIL` :

| Question | Refusée sur | Raison du refus |
|---|---|---|
| INRS-DEV-002 | 3 runs sur 6 | `nombre_non_cite_reponse` : la réponse contient un nombre absent des extraits vérifiés |
| INRS-DEV-005 | 1 run sur 6 | `nombre_non_cite_reponse` |
| INRS-DEV-016 | 1 run sur 6 | `extrait_introuvable` (une affirmation cite un extrait qui n'est pas mot pour mot dans son passage) |

La variabilité vient du LLM, non déterministe même à température 0. **Elle va toujours dans le sens de la prudence** : quand la sortie varie, c'est le garde-fou 2 qui refuse une réponse mal étayée. Aucun piège n'a changé de statut, et aucune variation n'a produit de réponse dangereuse. Le coût de cette prudence se lit dans le FRR, qui varie de 0,138 à 0,241.

Refus stables (6 runs sur 6) :

- **INRS-DEV-020** : garde-fou, `extrait_introuvable`.
- **INRS-DEV-022** : refus du LLM. Le passage attendu n'est pas parmi les 8 transmis (rang 11 en recherche).
- **INRS-DEV-032** et **INRS-DEV-034** : refus du LLM alors que le passage attendu est transmis. Le LLM juge l'information absente faute de formulation explicite, ce qui est la conséquence voulue d'une définition stricte de `info_presente`.

INRS-DEV-018 est répondue sur les six runs en citant un autre passage que le passage attendu. Cet autre passage est correct mais partiel : c'est la seule valide répondue sans la bonne source.

## Limite connue : INRS-DEV-048

INRS-DEV-048 (`out_of_scope`) est une **réponse dangereuse sur tous les runs, v1 comme v2**. La question demande quel accessoire sert à condamner un objet précis, alors que le guide ne le dit que par une photo. Le texte n'en contient que la légende, qui nomme l'objet sans décrire l'accessoire.

Le LLM répond en citant mot pour mot cette légende. La citation existe, elle est exacte et elle nomme bien l'objet demandé, donc le garde-fou 2 la valide. Mais l'**affirmation** (l'accessoire utilisé) n'est pas contenue dans l'extrait : elle est déduite, ou tirée de connaissances générales. La réponse ne contient pas de nombre, donc la parade sur les nombres ne s'applique pas.

C'est la limite assumée dans l'ADR 0001 : **une citation exacte prouve que l'extrait existe, pas qu'il établit l'affirmation.** La règle 8 du prompt v2 (« la réponse ne contient rien qui ne soit pas dans un extrait cité ») ne suffit pas, puisqu'elle n'est pas vérifiée.

**Piste v3 : un vérificateur d'implication.** Il contrôlerait que chaque affirmation, et la réponse, est impliquée par son extrait cité (modèle d'inférence textuelle ou LLM juge contraint, à mesurer). Il ne fait pas partie du système évalué. Une v3 serait réglée sur le jeu dev, puis évaluée sur un **nouveau** jeu test (ADR 0003).

## Décisions

| Décision | Référence |
|---|---|
| Double garde-fou (refus calibré, vérification mot pour mot des citations), et sa mise à jour : garde-fou 1 suspendu, refus confié au LLM et au garde-fou 2 | [ADR 0001](adr/0001-double-garde-fou.md) |
| Retrait du reranker FlashRank (scores saturés en français, aucune séparation des pièges), ni reranker ni similarité comme seuil de refus | [ADR 0002](adr/0002-retrait-reranker-flashrank.md) |
| Protocole de l'évaluation finale sur le jeu test, fixé avant toute utilisation du jeu test | [ADR 0003](adr/0003-protocole-eval-finale.md) |

Autres réglages, tous faits sur le jeu dev :

- **Poids de recherche (1, 0, 2) et 20 candidats :** grille fixée à l'avance. Le plein texte ajoutait du bruit, R@20 = 29/29.
- **8 passages transmis :** R@8 = 0,97.
- **Passage de la v1 à la v2 :** sur diagnostic des refus de la v1, sans assouplir `info_presente`.
