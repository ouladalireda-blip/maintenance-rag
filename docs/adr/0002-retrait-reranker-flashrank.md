# ADR 0002 — Retrait du reranker FlashRank ms-marco-MultiBERT-L-12

- Statut : accepté
- Date : 2026-11-10
- Modifie : ADR 0001 (garde-fou 1)

## Contexte

L'ADR 0001 prévoyait un reranking par cross-encoder après la recherche hybride, et un garde-fou 1 fondé sur son score : refus sans appel au LLM si le meilleur score est inférieur à un seuil τ calibré sur `evals/questions_dev.yaml`.

Le reranker retenu était FlashRank `ms-marco-MultiBERT-L-12` (BERT multilingue, local, sans appel réseau), noté sur `section_title + "\n" + content` avec une fenêtre de 512 tokens. Il a été mesuré sur le jeu dev (50 questions : 29 valides, 11 `near_miss`, 6 `identifier_trap`, 4 `out_of_scope`), derrière la recherche retenue (poids 1 / 0 / 2, 20 candidats, R@20 = 29/29).

Résultats : `evals/results/rerank_dev_2026-11-10_f2ffa9b.json` (`run_eval.py --mode rerank`).

### Qualité du classement (29 questions valides, rangs parmi les 20 candidats)

| | R@1 | R@3 | MRR |
|---|---|---|---|
| Avant reranking | 0,55 (16/29) | 0,79 (23/29) | 0,704 |
| Après reranking | 0,03 (1/29) | 0,24 (7/29) | 0,205 |

- Comparaison appariée après contre avant : **2 victoires, 26 défaites, 1 égalité**.
- 21 questions sur 29 voient le rang du bon passage changer de plus de 2 places. Une seule s'améliore nettement (INRS-DEV-022 : 11 → 2) ; les autres reculent, jusqu'à 1 → 18.

### Séparation des pièges : nulle

Score maximal du reranker par question (min / médiane / max) :

| Groupe | n | min | médiane | max |
|---|---|---|---|---|
| valid | 29 | 0,9990 | 0,9995 | 0,9997 |
| near_miss | 11 | 0,9990 | 0,9995 | 0,9997 |
| identifier_trap | 6 | 0,9991 | 0,9993 | 0,9994 |
| out_of_scope | 4 | 0,9993 | 0,9994 | 0,9996 |

**20 pièges sur 21** ont un score maximal supérieur au plus petit score maximal d'une question valide (`near_miss` 10/11, `identifier_trap` 6/6, `out_of_scope` 4/4). Tous les scores tiennent dans un intervalle de 7 × 10⁻⁴ : aucun τ ne sépare les groupes.

### Latence

Reranking de 20 passages par question sur le poste de développement (CPU) : **médiane 2,8 s**, p95 3,5 s.

### Cause : saturation en français

Le modèle a été sondé sur des phrases inventées, hors corpus :

| Question | Passage | Score |
|---|---|---|
| française (consigner un disjoncteur) | français pertinent | 0,999 |
| française | **français hors sujet (recette de carbonara)** | **0,93** |
| anglaise (même sens) | anglais pertinent | 0,99 |
| anglaise | anglais hors sujet (carbonara) | 0,0007 |

En anglais, le modèle discrimine. En français, il note haut tout passage français, même hors sujet. Sur un corpus entièrement français et d'un seul domaine (consignation), tous les candidats saturent vers 0,999 : l'ordre se joue sur des écarts de l'ordre de 10⁻⁴, c'est du bruit.

### Défaut supplémentaire : le score dépend du lot

Le même couple question-passage ne reçoit pas le même score selon les autres passages notés dans le même appel (remplissage au passage le plus long). Exemple : un passage noté 0,98 dans un lot de cinq obtient 0,87 noté seul. Un seuil τ appliqué à ce score dépendrait donc des autres candidats, ce qui le rend inutilisable comme critère de refus.

## Décision

- Le reranker FlashRank `ms-marco-MultiBERT-L-12` est **retiré du pipeline** : `rerank.enabled: false` dans `config/config.yaml`. Quand `enabled` vaut `false`, le pipeline ne charge pas le modèle ; l'ordre des passages est celui de la recherche hybride retenue.
- L'interface `Reranker` (`src/maintenance_rag/rerank.py`) et le mode d'évaluation `run_eval.py --mode rerank` sont **conservés** pour tester plus tard un autre modèle avec la même mesure. Ce mode reste le seul endroit qui charge le modèle, et il signale que `enabled` vaut `false`.
- Aucun autre modèle n'est adopté ici : pas de nouvelle dépendance, aucun téléchargement.

## Conséquences

### Garde-fou 1

Le garde-fou 1 de l'ADR 0001 (seuil τ sur le score de reranking) est **suspendu** : il n'y a plus de score à seuiller, et `guardrails.tau` reste `null`.

Un score de recherche ne le remplace pas. La similarité cosinus du meilleur candidat a été mesurée de la même façon (`evals/results/signals_dev_2026-11-10_f2ffa9b.json`, `--mode signals`). La plus petite similarité d'une question valide est 0,432. Au-dessus de cette valeur se trouvent 9 `near_miss` sur 11, 6 `identifier_trap` sur 6 et 3 `out_of_scope` sur 4. C'était déjà la crainte de l'ADR 0001 (« les scores cosinus séparent mal pertinent et voisin »), elle est maintenant mesurée.

Une question `near_miss` porte par construction sur le sujet du corpus : ses meilleurs passages sont proches, et rien dans un score de recherche ne dit que la réponse précise est absente. **Un score de recherche ne peut pas détecter un `near_miss`.**

Ce rôle passe à l'étape suivante :

- **le LLM** reçoit les passages et doit pouvoir refuser explicitement dans sa sortie structurée (« la réponse ne figure pas dans les passages ») ;
- **le garde-fou 2** reste la barrière déterministe : toute affirmation doit s'appuyer sur un extrait cité mot pour mot et retrouvé dans le passage.

### Limites et suites

- **Ce que le garde-fou 2 ne vérifie pas.** Il vérifie qu'une citation existe, pas qu'elle répond à la question. Une réponse à un `near_miss` qui cite fidèlement un passage voisin passerait. Le taux de bonnes réponses de refus du LLM sur `near_miss`, `identifier_trap` et `out_of_scope` doit donc être mesuré de bout en bout sur le jeu dev avant tout figement, puis sur `questions_test.yaml`.
- **Coût.** Sans refus avant génération, chaque question appelle le LLM, y compris les pièges.
- **Nombre de passages donnés au LLM** : à choisir sur la recherche seule (`--mode signals`). R@5 = 0,90, R@8 = 0,97 et R@10 = 0,97 ; 8 passages donnent le même rappel que 10.
- **Documentation à mettre à jour.** Les docstrings de `pipeline.py` et `guardrails.py`, ainsi que le commentaire de `guardrails.tau`, décrivent encore un garde-fou 1 sur le score de reranking. Ils seront mis à jour avec l'implémentation du pipeline.
- **Réintroduction d'un reranker.** Elle passe par le mode `rerank`, avec les mêmes critères : comparaison appariée avant/après, séparation des pièges, latence. Elle impose aussi de vérifier que le score ne dépend pas du lot.

## Alternatives écartées

- **Garder FlashRank pour le seul classement** : il dégrade le classement (26 défaites sur 29) et coûte 2,8 s par question.
- **Garder FlashRank pour le seul score de refus** : 20 pièges sur 21 dépassent la plus petite valide ; le score dépend en outre du lot.
- **Utiliser les logits bruts au lieu du score sigmoïde** : c'est une transformation monotone, l'ordre et la séparation restent identiques.
- **Noter un passage à la fois** : supprimerait la dépendance au lot, mais pas la saturation, et multiplierait les appels.
- **Seuil sur la similarité cosinus du meilleur candidat** : mesuré, il ne sépare pas non plus les pièges (18 sur 21 au-dessus de la plus petite valide).
- **Adopter tout de suite un autre cross-encoder multilingue** : nouvelle dépendance et téléchargement, hors du périmètre de cette décision. L'interface `Reranker` et le mode `rerank` permettront de le mesurer plus tard, sans changer le pipeline.
