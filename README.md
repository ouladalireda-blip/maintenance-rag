# maintenance-rag

Assistant documentaire de maintenance industrielle (RAG) : recherche hybride (vecteur + plein texte + références exactes), reranking multilingue, réponses avec citations vérifiées mot pour mot, et refus calibré quand le corpus ne contient pas la réponse.

## Démarrage

```sh
uv sync                      # Python 3.12, voir .python-version
cp .env.example .env         # puis renseigner les clés localement
```

Le PDF INRS n'est pas versionné : voir [docs/corpus.md](docs/corpus.md).

## Qualité

```sh
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Repères

- Configuration : [config/config.yaml](config/config.yaml)
- Schéma de base : [supabase/migrations/0001_init.sql](supabase/migrations/0001_init.sql)
- Corpus : [corpus/manifest.yaml](corpus/manifest.yaml), [docs/corpus.md](docs/corpus.md)
- Évaluation : [evals/](evals/)
- Décisions : [docs/adr/](docs/adr/)
