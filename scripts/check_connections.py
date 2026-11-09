"""Diagnostic des connexions externes (outil de développement, hors du package).

Lit .env et vérifie la base Supabase (schéma de 0001_init.sql) puis l'API d'embedding OpenAI.
N'affiche jamais de secret : clés et mot de passe sont masqués, y compris dans les messages
d'erreur. Code de sortie : 0 si tous les points sont OK, 1 sinon.

Usage :
    uv run python scripts/check_connections.py

Coût : un appel d'embedding sur un texte de quelques jetons (fraction de centime).
"""

import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "config.yaml"
SQL_DIMENSION = 1536  # vector(1536) dans supabase/migrations/0001_init.sql
EMBEDDING_PROBE = "test de connexion"
TABLES = ("documents", "chunks")


class Settings(BaseSettings):
    """Variables lues dans .env (ou l'environnement) ; seules celles utiles ici."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    supabase_db_url: SecretStr | None = None
    openai_api_key: SecretStr | None = None


class Report:
    def __init__(self, secrets: list[str]) -> None:
        self.failures = 0
        self.successes = 0
        self._secrets = [s for s in secrets if s]

    def scrub(self, text: str) -> str:
        """Retire toute valeur secrète d'un message (les erreurs peuvent citer l'URL)."""
        for secret in self._secrets:
            text = text.replace(secret, "****")
        return text

    def ok(self, message: str) -> None:
        self.successes += 1
        print(f"  OK     {self.scrub(message)}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"  ÉCHEC  {self.scrub(message)}")

    def info(self, message: str) -> None:
        print(f"  INFO   {self.scrub(message)}")

    def check(self, condition: bool, success: str, failure: str) -> bool:
        (self.ok if condition else self.fail)(success if condition else failure)
        return condition

    def error(self, step: str, exc: BaseException) -> None:
        self.fail(f"{step} : {type(exc).__name__} : {str(exc).strip().splitlines()[0]}")


def mask_key(key: str) -> str:
    return f"{key[:3]}… ({len(key)} caractères)" if len(key) > 8 else "****"


MALFORMED_URL = (
    "SUPABASE_DB_URL mal formée : caractère spécial non encodé dans le mot de passe ? "
    "(/, @, :, #, ?)"
)
VALID_SCHEMES = ("postgresql", "postgres")  # libpq accepte les deux


def url_problems(url: str) -> list[str]:
    """Contrôles de forme de l'URL, sans jamais lever d'exception ni citer l'URL.

    Un caractère spécial non encodé dans le mot de passe (#, ?, /) coupe l'URL au mauvais
    endroit : le reste du mot de passe devient un port non numérique (ValueError), un fragment,
    ou un second « @ ». Ces cas sont tous signalés par le même message.
    """
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port  # .port lève ValueError s'il n'est pas numérique
    except ValueError:
        return [MALFORMED_URL]
    if parts.fragment or parts.netloc.count("@") > 1:
        return [MALFORMED_URL]
    problems = []
    if parts.scheme not in VALID_SCHEMES:
        problems.append("SUPABASE_DB_URL : schéma invalide, elle doit commencer par postgresql://")
    if not host:
        problems.append("SUPABASE_DB_URL : hôte absent")
    if port is None:
        problems.append("SUPABASE_DB_URL : port absent (attendu numérique, ex. 5432 ou 6543)")
    return problems


def url_password(url: str) -> str:
    """Mot de passe de l'URL, ou chaîne vide si elle n'est pas analysable."""
    try:
        return urlsplit(url).password or ""
    except ValueError:
        return ""


def mask_url(url: str) -> str:
    """URL affichable avec le mot de passe masqué ; ne lève jamais d'exception."""
    try:
        parts = urlsplit(url)
        user = parts.username or ""
        host = parts.hostname or "?"
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return "(non affichée : URL non analysable)"
    credentials = f"{user}:****@" if parts.password else (f"{user}@" if user else "")
    return f"{parts.scheme}://{credentials}{host}{port}{parts.path}"


# ---------------------------------------------------------------------------------------------
def check_database(url: str, report: Report) -> None:
    import psycopg
    from psycopg import sql

    problems = url_problems(url)
    if problems:  # URL non conforme : aucune tentative de connexion, URL jamais affichée
        print("\n== Base de données (SUPABASE_DB_URL non affichée)")
        for problem in problems:
            report.fail(problem)
        return

    print(f"\n== Base de données (SUPABASE_DB_URL = {mask_url(url)})")
    try:
        # prepare_threshold=None : compatible avec le pooler Supabase en mode transaction.
        connection = psycopg.connect(url, connect_timeout=10, prepare_threshold=None)
    except Exception as exc:  # diagnostic : toute erreur devient un échec affiché
        report.error("Connexion", exc)
        return

    with connection, connection.cursor() as cursor:

        def one(query: str, *params: Any) -> Any:
            cursor.execute(query, params or None)
            row = cursor.fetchone()
            return row[0] if row else None

        try:
            version = one("show server_version")
            report.ok(f"Connexion (PostgreSQL {version})")

            extension = one("select extversion from pg_extension where extname = 'vector'")
            report.check(
                extension is not None,
                f"Extension vector installée (version {extension})",
                "Extension vector absente : appliquer 0001_init.sql",
            )

            cursor.execute(
                """
                select c.relname, c.relrowsecurity
                from pg_class c join pg_namespace n on n.oid = c.relnamespace
                where n.nspname = 'public' and c.relkind = 'r' and c.relname = any(%s)
                """,
                (list(TABLES),),
            )
            found = dict(cursor.fetchall())
            for table in TABLES:
                if not report.check(
                    table in found, f"Table {table} présente", f"Table {table} absente"
                ):
                    continue
                report.check(
                    bool(found[table]),
                    f"RLS activée sur {table}",
                    f"RLS DÉSACTIVÉE sur {table} : données exposées à la clé publique",
                )
                count = one(sql.SQL("select count(*) from public.{}").format(sql.Identifier(table)))
                report.info(f"{table} : {count} ligne(s)")

            if "chunks" in found:
                column_type = one(
                    """
                    select format_type(a.atttypid, a.atttypmod)
                    from pg_attribute a
                    where a.attrelid = 'public.chunks'::regclass and a.attname = 'embedding'
                    """
                )
                report.check(
                    column_type is not None and column_type.endswith(f"vector({SQL_DIMENSION})"),
                    f"Colonne chunks.embedding : {column_type}",
                    f"Colonne chunks.embedding : {column_type} (attendu vector({SQL_DIMENSION}))",
                )

            arguments = one(
                """
                select pg_get_function_identity_arguments(p.oid)
                from pg_proc p join pg_namespace n on n.oid = p.pronamespace
                where n.nspname = 'public' and p.proname = 'hybrid_search'
                """
            )
            report.check(
                arguments is not None,
                "Fonction hybrid_search présente",
                "Fonction hybrid_search absente",
            )
        except Exception as exc:
            report.error("Requête de vérification", exc)


def check_openai(key: str, model: str, expected_dimension: int, report: Report) -> None:
    from openai import OpenAI

    print(f"\n== OpenAI (OPENAI_API_KEY = {mask_key(key)}, modèle {model})")
    report.check(
        expected_dimension == SQL_DIMENSION,
        f"config.yaml embeddings.dimensions = {expected_dimension} (= SQL)",
        f"config.yaml embeddings.dimensions = {expected_dimension} ≠ SQL {SQL_DIMENSION}",
    )
    try:
        client = OpenAI(api_key=key, timeout=30.0, max_retries=1)
        response = client.embeddings.create(model=model, input=EMBEDDING_PROBE)
    except Exception as exc:
        report.error("Appel d'embedding", exc)
        return
    dimension = len(response.data[0].embedding)
    report.check(
        dimension == SQL_DIMENSION,
        f"Embedding reçu : dimension {dimension} (attendue {SQL_DIMENSION})",
        f"Embedding reçu : dimension {dimension} ≠ {SQL_DIMENSION} attendue par le schéma SQL",
    )
    report.info(f"Jetons consommés : {response.usage.total_tokens} (modèle {response.model})")


# ---------------------------------------------------------------------------------------------
def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    settings = Settings()
    url = settings.supabase_db_url.get_secret_value() if settings.supabase_db_url else ""
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
    password = url_password(url)
    report = Report(secrets=[url, key, password, unquote(password)])

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    embeddings = config["embeddings"]

    if url:
        check_database(url, report)
    else:
        print("\n== Base de données")
        report.fail("SUPABASE_DB_URL absente de .env")

    if key:
        check_openai(key, embeddings["model"], int(embeddings["dimensions"]), report)
    else:
        print("\n== OpenAI")
        report.fail("OPENAI_API_KEY absente de .env")

    verdict = "TOUT EST OK" if report.failures == 0 else "DES POINTS ÉCHOUENT"
    print(f"\nRésultat : {report.successes} OK, {report.failures} ÉCHEC — {verdict}")
    return 0 if report.failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
