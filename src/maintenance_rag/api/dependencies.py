"""État de l'API, créé UNE fois au démarrage (lifespan), et dépendances FastAPI.

- ``ApiSettings`` : section ``api`` de config/config.yaml (budget journalier, origines CORS,
  chemin du journal).
- ``BudgetGuard`` : garde-fou de coût journalier, sûr en cas de requêtes simultanées (voir sa
  docstring).
- ``AppState`` : le pipeline (``answerer``), la vérification de disponibilité (``ready``), le
  journal, le garde-fou de budget et les titres des documents. Les tests le remplacent par un
  état factice via ``app.dependency_overrides[get_state]``.
- ``build_state`` : construit l'état réel. Le ``Pipeline`` (recherche, prompt, garde-fou
  inchangés) est créé une fois ; la base n'est PAS partagée entre requêtes : chaque recherche,
  et chaque contrôle /ready, ouvre sa propre connexion en lecture seule puis la ferme.
  L'embedder est l'embedder OpenAI direct, sans ``CachedEmbedder`` (qui réécrit un fichier JSON
  à chaque question et n'est pas sûr avec des requêtes simultanées). Les secrets viennent de
  l'environnement / .env via ``Settings`` ; ils ne sont ni journalisés ni renvoyés. Si la
  configuration est incomplète au démarrage, l'API démarre quand même : /health répond,
  /ready et /ask renvoient 503.
"""

import logging
import ssl
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import openai
import psycopg
import yaml
from fastapi import Request

from maintenance_rag.config import CONFIG_PATH, ROOT, load_manifest
from maintenance_rag.pipeline import PipelineResult

from .logging import RequestLog

logger = logging.getLogger("maintenance_rag.api")

# Erreurs « en amont » : réseau, TLS, OpenAI, base. Elles donnent 503, jamais un refus.
UPSTREAM_ERRORS: tuple[type[BaseException], ...] = (
    openai.OpenAIError,
    psycopg.Error,
    ssl.SSLError,
    ConnectionError,
    TimeoutError,
)

# Majorant des tokens d'entrée d'une question : prompt système + schéma (~1 500), 8 passages de
# 350 tokens au plus (2 800), question de 500 caractères au plus. Sert à réserver le coût
# maximal d'une question avant l'appel au LLM.
MAX_INPUT_TOKENS = 8_000


class Answerer(Protocol):
    def answer(self, question: str) -> PipelineResult: ...


@dataclass(frozen=True)
class ApiSettings:
    daily_budget_usd: float
    cors_origins: list[str]
    log_path: Path

    @classmethod
    def from_config(cls, config_path: Path = CONFIG_PATH) -> "ApiSettings":
        api = yaml.safe_load(config_path.read_text(encoding="utf-8")).get("api") or {}
        log_path = Path(api.get("log_path", "data/api_log.sqlite"))
        settings = cls(
            daily_budget_usd=float(api.get("daily_budget_usd", 1.0)),
            cors_origins=[str(origin) for origin in api.get("cors_origins", [])],
            log_path=log_path if log_path.is_absolute() else ROOT / log_path,
        )
        if settings.daily_budget_usd < 0:
            raise ValueError("api.daily_budget_usd doit être positif ou nul")
        return settings


class BudgetGuard:
    """Garde-fou de coût journalier, sûr en cas de requêtes simultanées.

    Avant l'appel au LLM, chaque requête RÉSERVE le coût maximal d'une question
    (``reservation_usd``). La réservation est refusée si coût du jour déjà journalisé +
    réservations en cours + cette réservation dépasse le budget : des requêtes simultanées ne
    peuvent donc pas, ensemble, dépasser le budget. Après l'appel, ``settle`` journalise le coût
    réel et libère la réservation, sous le même verrou.
    """

    def __init__(self, log: RequestLog, daily_budget_usd: float, reservation_usd: float) -> None:
        if reservation_usd < 0:
            raise ValueError("réservation négative")
        self.log = log
        self.daily_budget_usd = daily_budget_usd
        self.reservation_usd = reservation_usd
        self._reserved = 0.0
        self._lock = threading.Lock()

    def reserve(self) -> bool:
        with self._lock:
            committed = self.log.spent_today() + self._reserved
            if committed >= self.daily_budget_usd:
                return False
            if committed + self.reservation_usd > self.daily_budget_usd:
                return False
            self._reserved += self.reservation_usd
            return True

    def settle(self, record: Callable[[], None]) -> None:
        """Journalise (``record``) puis libère la réservation, de façon atomique."""
        with self._lock:
            try:
                record()
            finally:
                self._reserved = max(0.0, self._reserved - self.reservation_usd)

    @property
    def reserved(self) -> float:
        with self._lock:
            return self._reserved


@dataclass
class ReadyStatus:
    database: bool
    llm_config: bool
    prompt_version: str | None

    @property
    def ready(self) -> bool:
        return self.database and self.llm_config


@dataclass
class AppState:
    answerer: Answerer | None
    ready: Callable[[], ReadyStatus]
    log: RequestLog
    settings: ApiSettings
    budget: BudgetGuard
    documents: dict[str, str] = field(default_factory=dict)  # slug → titre


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.rag
    return state


def _document_titles() -> dict[str, str]:
    try:
        return {entry.id: entry.title for entry in load_manifest()}
    except (OSError, ValueError):
        return {}


def build_state() -> AppState:
    """État réel. Aucun secret n'est journalisé : seuls des types d'erreur le sont."""
    from maintenance_rag.embeddings import OpenAIEmbedder
    from maintenance_rag.generation import PROMPT_VERSION
    from maintenance_rag.ingestion import Settings
    from maintenance_rag.llm import OpenAILLM
    from maintenance_rag.pipeline import Pipeline, PipelineSettings
    from maintenance_rag.retrieval import SearchResult, search

    api_settings = ApiSettings.from_config()
    log = RequestLog(api_settings.log_path)
    documents = _document_titles()

    def degraded(reason: str) -> AppState:
        logger.error("pipeline indisponible au démarrage (%s)", reason)
        return AppState(
            answerer=None,
            ready=lambda: ReadyStatus(database=False, llm_config=False, prompt_version=None),
            log=log,
            settings=api_settings,
            budget=BudgetGuard(log, api_settings.daily_budget_usd, 0.0),
            documents=documents,
        )

    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        pipeline_settings = PipelineSettings.from_config()
        llm_settings = pipeline_settings.llm
        if not llm_settings.model:
            return degraded("llm.model vide")
        if llm_settings.price_per_1m_input is None or llm_settings.price_per_1m_output is None:
            return degraded("prix du LLM non renseignés : budget impossible à suivre")
        secrets = Settings()  # environnement / .env ; jamais affiché
        database_url = secrets.supabase_db_url.get_secret_value()
        api_key = secrets.openai_api_key.get_secret_value()
        embeddings = config["embeddings"]
        embedder = OpenAIEmbedder(  # embedder direct : pas de fichier cache partagé
            model=embeddings["model"],
            dimensions=int(embeddings["dimensions"]),
            api_key=api_key,
            batch_size=int(embeddings["batch_size"]),
            max_attempts=int(embeddings["max_attempts"]),
        )
        llm = OpenAILLM(llm_settings, api_key=api_key)
        reranker = None
        if pipeline_settings.rerank_enabled:
            from maintenance_rag.rerank import FlashRankReranker

            reranker = FlashRankReranker.from_config()
    except Exception as error:  # configuration, secrets absents : démarrage dégradé
        return degraded(type(error).__name__)

    def connect() -> psycopg.Connection[tuple[object, ...]]:
        """Une connexion par usage, en lecture seule, jamais partagée entre requêtes."""
        conn = psycopg.connect(database_url, autocommit=True)
        conn.read_only = True
        return conn

    def retrieve(question: str) -> Sequence[SearchResult]:
        with connect() as conn:
            return search(question, embedder, conn)

    pipeline = Pipeline(retrieve, llm, pipeline_settings, reranker)

    def ready() -> ReadyStatus:
        database = False
        try:
            with connect() as conn, conn.cursor() as cursor:
                cursor.execute("select 1")
                database = cursor.fetchone() == (1,)
        except UPSTREAM_ERRORS as error:
            logger.warning("base indisponible (%s)", type(error).__name__)
        return ReadyStatus(database=database, llm_config=True, prompt_version=PROMPT_VERSION)

    reservation = llm_settings.cost(MAX_INPUT_TOKENS, llm_settings.max_output_tokens) or 0.0
    return AppState(
        answerer=pipeline,
        ready=ready,
        log=log,
        settings=api_settings,
        budget=BudgetGuard(log, api_settings.daily_budget_usd, reservation),
        documents=documents,
    )
