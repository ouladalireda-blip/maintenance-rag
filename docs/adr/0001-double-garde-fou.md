# ADR 0001 — Double garde-fou : refus calibré et vérification des citations

- Statut : accepté
- Date : 2026-11-09

## Contexte

L'assistant répond à des questions de maintenance, dont des procédures de consignation : une réponse inventée ou mal sourcée peut exposer un technicien à un danger. Deux défaillances sont à exclure :

1. **Répondre alors que le corpus ne contient pas la réponse** (question hors sujet, voisine du corpus, ou piège sur une référence inexistante).
2. **Citer un extrait qui ne figure pas dans la source**, ou le déformer, alors même que la recherche a trouvé les bons passages.

Un LLM seul ne garantit ni l'un ni l'autre : les consignes du type « réponds “je ne sais pas” si… » sont suivies de manière irrégulière, et le modèle peut paraphraser une citation présentée comme littérale.

## Décision

Deux garde-fous indépendants, l'un avant, l'autre après la génération :

- **Garde-fou 1 — refus avant génération.** Si le meilleur score de reranking (cross-encoder) est **inférieur à τ**, le système refuse sans appeler le LLM. τ est **calibré sur `evals/questions_dev.yaml`** (compromis entre refus à tort sur `valid` et réponses à tort sur `out_of_scope`, `near_miss` et `identifier_trap`), puis **figé** dans `config/config.yaml` avant toute mesure sur `questions_test.yaml`. Tant qu'il n'est pas calibré, `tau: null`.
- **Garde-fou 2 — vérification après génération.** Le LLM produit une **réponse structurée** (réponse + liste d'extraits cités, chacun rattaché à un passage). Chaque extrait est **vérifié de façon déterministe, mot pour mot**, contre le texte du passage cité (après une normalisation documentée des espaces et de la typographie). Un extrait introuvable invalide la réponse, qui devient un refus.

## Alternatives écartées

- **Consigne de refus dans le prompt seule** : non mesurable, non déterministe, contournée par les questions proches du corpus.
- **Seuil sur la similarité vectorielle** : les scores cosinus séparent mal « pertinent » et « voisin » ; le cross-encoder, qui lit question et passage ensemble, discrimine mieux.
- **Seuil τ fixé à la main** : arbitraire et non justifiable ; la calibration sur un jeu dédié le rend traçable.
- **Vérification des citations par un LLM juge** : coûteuse, non déterministe, et elle-même sujette à erreur.
- **Correspondance approximative (fuzzy) des extraits** : tolère précisément les déformations qu'on veut détecter (chiffre, négation, unité).

## Conséquences

- Le système refusera parfois des questions auxquelles il aurait su répondre : ce taux de faux refus est mesuré et assumé.
- L'évaluation sépare `dev` (calibration) et `test` (mesure finale, figé et commité avant la mesure).
- La génération doit utiliser une sortie structurée, et les passages conservent leur texte exact et leurs pages jusqu'à la vérification.
- Changer de modèle de reranking impose de recalibrer τ.

## Mise à jour (ADR 0002) — 2026-11-11

- **Garde-fou 1 suspendu.** Ni le score du reranker FlashRank (saturé en français) ni la similarité cosinus du meilleur candidat ne séparent les pièges des questions valides sur le jeu dev (`docs/adr/0002-retrait-reranker-flashrank.md`). Il n'y a donc plus de seuil τ : `guardrails.tau` est retiré de `config/config.yaml`.
- **Le refus repose sur le LLM et le garde-fou 2.** Le LLM reçoit les `generation.top_k` (8) meilleurs passages de la recherche. Sa sortie structurée porte `info_presente` : faux si l'information précise demandée n'est pas écrite dans les passages, ce qui donne le statut `REFUSED_BY_LLM`. Le garde-fou 2 (`guardrails.verify`) vérifie ensuite, sans LLM, chaque affirmation : passage fourni, extrait mot pour mot (au moins 20 caractères), nombres cités. Une seule affirmation fausse refuse toute la réponse (`REFUSED_BY_GUARDRAIL`).
- **Limite assumée.** Une citation exacte prouve que l'extrait existe, pas qu'il répond à la question. Une réponse à une question voisine (`near_miss`) qui cite fidèlement un passage voisin passe le garde-fou 2.
- **Parades.**
  1. Tout nombre d'une affirmation (valeur, durée, seuil, article) doit figurer dans son extrait. Un chiffre inventé ou pris ailleurs fait échouer la vérification.
  2. Les réponses dangereuses (pièges en `ANSWERED`) sont mesurées de bout en bout (`run_eval.py --mode e2e`), contre des critères fixés avant la première mesure : au plus 1 réponse dangereuse sur 21 pièges, FRR ≤ 20 %, bonne source ≥ 80 % des valides répondues.
