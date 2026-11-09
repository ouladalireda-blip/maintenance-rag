# Image de l'API. Le corpus n'est PAS copié dans l'image (corpus/inrs/ est protégé) :
# il est monté en lecture seule par docker-compose. Les secrets viennent de l'environnement.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances d'abord (cache de couche), puis le code. Copie explicite : ni .env ni corpus/.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY config ./config
RUN uv sync --locked --no-dev

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "maintenance_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
