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

## Lancer avec Docker

L'API (FastAPI) et l'interface (Streamlit, cliente de l'API) tournent dans deux conteneurs.

**Prérequis**

- Docker avec Docker Compose v2 (Docker Desktop sous Windows ou macOS), démarré.
- Une base PostgreSQL / Supabase déjà alimentée par l'ingestion : le texte du guide est en base, le PDF n'est pas nécessaire à l'exécution.

**Fichier `.env`** (à la racine, jamais versionné ni copié dans une image) : copier `.env.example` en `.env` et renseigner les valeurs localement.

| Variable | Rôle |
|---|---|
| `OPENAI_API_KEY` | embeddings des questions et génération (nécessaire à l'API) |
| `SUPABASE_DB_URL` | connexion PostgreSQL, en lecture seule (nécessaire à l'API) |
| `ANTHROPIC_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | présentes dans `.env.example`, non utilisées par l'API |

Sans `.env` valide, l'API démarre en mode dégradé : `/health` répond 200, `/ready` et `/ask` répondent 503.

**Démarrer**

```sh
docker compose up -d --build
docker compose ps        # les deux services doivent être « healthy »
```

L'interface attend que l'API soit « healthy » avant de démarrer. Le premier contrôle de santé de l'API a lieu au plus tard 90 s après son démarrage.

**Adresses**

- Interface : http://localhost:8600
- API (documentation interactive) : http://localhost:8100/docs

Les ports 8100 et 8600 évitent les conflits avec un autre projet local sur 8000 et 8501.

**Données conservées.** Le journal des questions et les avis « utile / pas utile » sont stockés dans le volume nommé `api-data`, qui survit aux redémarrages.

**Derrière un antivirus ou un proxy qui inspecte HTTPS.** Si la construction échoue avec `invalid peer certificate: UnknownIssuer` (ou `CERTIFICATE_VERIFY_FAILED`), un logiciel du poste remplace les certificats des sites par les siens. Les images peuvent faire confiance à son autorité racine, **en plus** des autorités publiques, sans désactiver aucune vérification :

1. Identifier l'émetteur depuis un conteneur :
   ```sh
   docker run --rm python:3.12-slim sh -c "openssl s_client -connect pypi.org:443 -servername pypi.org </dev/null 2>/dev/null | grep -E 'issuer|depth'"
   ```
   Un émetteur qui n'est pas une autorité publique (par exemple `CN=Norton Web/Mail Shield Root`) signale une inspection.
2. Exporter le certificat **racine public** de cet émetteur au format PEM (depuis le magasin de certificats du système ou les réglages du logiciel) dans `certs/`, par exemple `certs/inspection-ca.pem`. Jamais de clé privée : la construction refuse un tel fichier.
3. Reconstruire : `docker compose up -d --build`. Le certificat est ajouté au magasin système de l'image (`update-ca-certificates`), utilisé par uv au build et par Python à l'exécution.

Règle : **ce fichier ne doit jamais être versionné** (propre au poste ; `.gitignore` n'autorise que `certs/.keep`, et la CI le vérifie). Sans certificat dans `certs/` (CI, autre poste), la construction fonctionne telle quelle.

**Arrêter**

```sh
docker compose down        # arrête et supprime les conteneurs ; le volume api-data est conservé
docker compose down -v     # supprime aussi le volume (journal et avis perdus)
```

## Repères

- Configuration : [config/config.yaml](config/config.yaml)
- Schéma de base : [supabase/migrations/0001_init.sql](supabase/migrations/0001_init.sql)
- Corpus : [corpus/manifest.yaml](corpus/manifest.yaml), [docs/corpus.md](docs/corpus.md)
- Évaluation : [evals/](evals/)
- Décisions : [docs/adr/](docs/adr/)
