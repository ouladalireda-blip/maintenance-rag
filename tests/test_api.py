"""Tests de l'API FastAPI avec un pipeline factice injecté (dependency override), sans réseau.

Le client de test est utilisé SANS bloc ``with`` (sauf pour tester le lifespan avec un état
factice) : le vrai pipeline, avec la base et OpenAI, n'est jamais construit.
"""

import datetime as dt
import inspect
import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import openai
import psycopg
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from maintenance_rag.api import dependencies
from maintenance_rag.api.dependencies import (
    ApiSettings,
    AppState,
    BudgetGuard,
    ReadyStatus,
    get_state,
)
from maintenance_rag.api.logging import RequestLog
from maintenance_rag.api.main import create_app
from maintenance_rag.api.schemas import AVERTISSEMENT
from maintenance_rag.llm import FakeLLM, LLMSettings
from maintenance_rag.pipeline import (
    NOT_FOUND_MESSAGE,
    REFUSAL_MESSAGE,
    Pipeline,
    PipelineResult,
    PipelineSettings,
)
from maintenance_rag.retrieval import SearchResult

CONTENT = "Le délai d'attente avant intervention est de 5 minutes après la coupure."
QUOTE = "Le délai d'attente avant intervention est de 5 minutes"
PASSAGE = SearchResult(
    chunk_id=100,
    document_slug="inrs-ed6109",
    chunk_index=0,
    section_title="5.1. Consignation électrique",
    page_start=16,
    page_end=17,
    printed_page_start=14,
    printed_page_end=15,
    semantic_rank=1,
    full_text_rank=None,
    identifier_rank=None,
    rrf_score=0.02,
    has_warning=False,
    content=CONTENT,
)
SETTINGS = PipelineSettings(
    top_k=8,
    min_quote_chars=20,
    rerank_enabled=False,
    llm=LLMSettings("openai", "fake", 0.0, 2000, 1.0, 10.0),
)
COST = (2000 * 1.0 + 300 * 10.0) / 1_000_000  # coût d'un appel au FakeLLM ci-dessous


def _json(present: bool, *affirmations: tuple[str, str, str], reponse: str = "Attendre.") -> str:
    return json.dumps(
        {
            "info_presente": present,
            "reponse": reponse,
            "affirmations": [
                {"texte": t, "extrait_exact": e, "passage_id": p} for t, e, p in affirmations
            ],
        }
    )


ANSWERED = _json(True, ("Attendre 5 minutes après la coupure.", QUOTE, "P1"))
REFUSED_LLM_RELATED = _json(
    False, ("Information liée : attendre 5 minutes.", QUOTE, "P1"), reponse="Absent du guide."
)
REFUSED_LLM_UNVERIFIED = _json(False, ("Information liée.", "phrase absente de tout passage", "P1"))
REFUSED_GUARDRAIL = _json(True, ("Attendre 10 minutes.", QUOTE, "P1"))  # nombre non cité


class Raising:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def answer(self, question: str) -> PipelineResult:
        raise self.error


Maker = Callable[..., tuple[TestClient, AppState, FakeLLM]]


@pytest.fixture
def make_client(tmp_path: Path) -> Maker:
    def make(
        llm_text: str = ANSWERED,
        *,
        answerer: Any = None,
        ready: ReadyStatus | None = None,
        budget: float = 1.0,
        reservation: float = COST,
    ) -> tuple[TestClient, AppState, FakeLLM]:
        llm = FakeLLM(lambda system, user: llm_text, input_tokens=2000, output_tokens=300)
        log = RequestLog(tmp_path / "api_log.sqlite")
        state = AppState(
            answerer=answerer or Pipeline(lambda question: [PASSAGE], llm, SETTINGS),
            ready=lambda: ready or ReadyStatus(True, True, "generation-v2-2026-11-11"),
            log=log,
            settings=ApiSettings(budget, ["http://localhost:5173"], log.path),
            budget=BudgetGuard(log, budget, reservation),
            documents={"inrs-ed6109": "INRS ED 6109 — Consignations et déconsignations"},
        )
        app = create_app(build=lambda: state)
        app.dependency_overrides[get_state] = lambda: state
        return TestClient(app), state, llm

    return make


def _ask(client: TestClient, question: str = "Combien de temps attendre après la coupure ?") -> Any:
    return client.post("/ask", json={"question": question})


def _rows(state: AppState) -> list[dict[str, Any]]:
    conn = sqlite3.connect(state.log.path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("select * from requests")]
    finally:
        conn.close()


# ------------------------------------------------------------------------------- /health, /ready
def test_health_makes_no_external_call(make_client: Maker) -> None:
    client, _, llm = make_client(answerer=Raising(AssertionError("aucun appel attendu")))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert llm.calls == []


def test_ready_ok(make_client: Maker) -> None:
    client, _, _ = make_client()
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "database": True,
        "llm_config": True,
        "prompt_version": "generation-v2-2026-11-11",
    }


def test_ready_503_without_secrets(make_client: Maker) -> None:
    client, _, _ = make_client(ready=ReadyStatus(False, True, "generation-v2-2026-11-11"))
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready" and body["database"] is False
    assert "postgres" not in response.text.lower() and "sk-" not in response.text


# ------------------------------------------------------------------------------------- /ask
def test_ask_answered(make_client: Maker) -> None:
    client, _, _ = make_client(ANSWERED)
    response = _ask(client)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ANSWERED"
    assert body["reponse"] == "Attendre."
    assert body["refus_raison"] is None
    assert body["citations"] == [
        {
            "texte": "Attendre 5 minutes après la coupure.",
            "extrait_exact": QUOTE,
            "page_imprimee": "14-15",
            "section": "5.1. Consignation électrique",
            "document": "INRS ED 6109 — Consignations et déconsignations",
        }
    ]
    assert body["avertissement"] == AVERTISSEMENT
    assert body["prompt_version"] == "generation-v2-2026-11-11"
    assert body["cout_usd"] == pytest.approx(COST)
    assert body["latence_ms"] >= 0 and len(body["request_id"]) == 32


def test_ask_refused_by_llm_with_verified_related_info(make_client: Maker) -> None:
    client, _, _ = make_client(REFUSED_LLM_RELATED)
    body = _ask(client).json()
    assert body["status"] == "REFUSED_BY_LLM"
    assert body["reponse"] == "Absent du guide."  # réponse du pipeline, inchangée
    assert body["refus_raison"] is None
    assert [c["extrait_exact"] for c in body["citations"]] == [QUOTE]  # toutes vérifiées


def test_ask_refused_by_llm_hides_unverified_related_info(make_client: Maker) -> None:
    client, _, _ = make_client(REFUSED_LLM_UNVERIFIED)
    body = _ask(client).json()
    assert body["status"] == "REFUSED_BY_LLM"
    assert body["reponse"] == NOT_FOUND_MESSAGE  # message standard du pipeline
    assert body["citations"] == []


def test_ask_refused_by_guardrail(make_client: Maker) -> None:
    client, _, _ = make_client(REFUSED_GUARDRAIL)
    body = _ask(client).json()
    assert body["status"] == "REFUSED_BY_GUARDRAIL"
    assert body["reponse"] == REFUSAL_MESSAGE  # message standard du pipeline
    assert body["refus_raison"] == "nombre_non_cite"
    assert body["citations"] == []


def test_warning_always_present(make_client: Maker) -> None:
    for text in (ANSWERED, REFUSED_LLM_RELATED, REFUSED_LLM_UNVERIFIED, REFUSED_GUARDRAIL):
        client, _, _ = make_client(text)
        assert _ask(client).json()["avertissement"] == AVERTISSEMENT


@pytest.mark.parametrize("question", ["", "   ", "ab", "  a  ", "x" * 501, "  " + "x" * 501 + "  "])
def test_ask_invalid_question_is_422(make_client: Maker, question: str) -> None:
    client, state, llm = make_client()
    assert _ask(client, question).status_code == 422
    assert llm.calls == [] and _rows(state) == []


def test_ask_rejects_missing_or_extra_fields(make_client: Maker) -> None:
    client, _, _ = make_client()
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": "Quelle procédure ?", "x": 1}).status_code == 422


def test_ask_question_is_stripped(make_client: Maker) -> None:
    client, state, llm = make_client()
    assert _ask(client, "   Quelle procédure ?   ").status_code == 200
    assert "Question : Quelle procédure ?\n" in llm.calls[0][1]
    assert _rows(state)[0]["question"] == "Quelle procédure ?"


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1")),
        openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1")),
        psycopg.OperationalError("connexion perdue"),
        ConnectionResetError("réseau"),
    ],
    ids=["openai-connexion", "openai-timeout", "base", "reseau"],
)
def test_ask_network_error_is_503_not_a_refusal(make_client: Maker, error: BaseException) -> None:
    client, state, _ = make_client(answerer=Raising(error))
    response = _ask(client)
    assert response.status_code == 503
    body = response.json()
    assert "status" not in body and "REFUSED" not in response.text
    assert "indisponible" in body["detail"] and body["request_id"]
    (row,) = _rows(state)
    assert row["status"] == "UPSTREAM_ERROR" and row["cost_usd"] == 0
    assert state.budget.reserved == 0  # réservation libérée


def test_ask_not_ready_is_503(make_client: Maker) -> None:
    client, state, _ = make_client()
    state.answerer = None
    assert _ask(client).status_code == 503


# ------------------------------------------------------------------------------- budget
def test_ask_over_daily_budget_is_429_without_llm_call(make_client: Maker) -> None:
    client, state, llm = make_client(budget=COST * 1.5)
    assert _ask(client).status_code == 200  # 1er appel : coût journalisé
    response = _ask(client)  # reste 0,5 × COST < une réservation
    assert response.status_code == 429
    assert "Budget journalier" in response.json()["detail"]
    assert len(llm.calls) == 1  # aucun nouvel appel au LLM
    assert len(_rows(state)) == 1  # le refus 429 n'est pas un coût


def test_budget_already_spent_today_is_429(make_client: Maker) -> None:
    client, state, llm = make_client(budget=1.0)
    state.log.record(
        request_id="avant",
        question="q",
        status="ANSWERED",
        refus_raison=None,
        cited_chunk_ids=[],
        latency_ms=1,
        cost_usd=1.0,
        prompt_version="v",
    )
    assert _ask(client).status_code == 429
    assert llm.calls == []


def test_budget_counts_only_today(make_client: Maker) -> None:
    client, state, _ = make_client(budget=1.0)
    state.log.record(
        request_id="hier",
        question="q",
        status="ANSWERED",
        refus_raison=None,
        cited_chunk_ids=[],
        latency_ms=1,
        cost_usd=5.0,
        prompt_version="v",
        now=dt.datetime.now().astimezone() - dt.timedelta(days=1),
    )
    assert state.log.spent_today() == 0
    assert _ask(client).status_code == 200


def test_budget_guard_is_safe_under_concurrency(tmp_path: Path) -> None:
    """20 réservations simultanées, budget pour 3 : exactement 3 acceptées."""
    guard = BudgetGuard(
        RequestLog(tmp_path / "log.sqlite"), daily_budget_usd=0.35, reservation_usd=0.1
    )
    barrier = threading.Barrier(20)
    results: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        accepted = guard.reserve()
        with lock:
            results.append(accepted)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 3
    assert guard.reserved == pytest.approx(0.3)


def test_budget_guard_settle_records_and_releases(tmp_path: Path) -> None:
    log = RequestLog(tmp_path / "log.sqlite")
    guard = BudgetGuard(log, daily_budget_usd=0.25, reservation_usd=0.1)
    assert guard.reserve() and guard.reserve() and not guard.reserve()
    guard.settle(
        lambda: log.record(
            request_id="r1",
            question="q",
            status="ANSWERED",
            refus_raison=None,
            cited_chunk_ids=[],
            latency_ms=1,
            cost_usd=0.02,
            prompt_version="v",
        )
    )
    assert guard.reserved == pytest.approx(0.1)
    assert log.spent_today() == pytest.approx(0.02)
    assert guard.reserve()  # 0,02 dépensé + 0,1 réservé + 0,1 ≤ 0,25


# ------------------------------------------------------------------------------- journal
def test_log_is_written_without_guide_text(make_client: Maker) -> None:
    client, state, _ = make_client(ANSWERED)
    body = _ask(client).json()
    (row,) = _rows(state)
    assert row["request_id"] == body["request_id"]
    assert row["question"] == "Combien de temps attendre après la coupure ?"
    assert row["status"] == "ANSWERED" and row["refus_raison"] is None
    assert json.loads(row["cited_chunk_ids"]) == [100]
    assert row["prompt_version"] == "generation-v2-2026-11-11"
    assert row["latency_ms"] >= 0 and row["cost_usd"] == pytest.approx(COST)
    assert row["ts"] and row["day"] == dt.date.today().isoformat()
    everything = " ".join(str(value) for value in row.values())
    assert "délai d'attente" not in everything  # aucun extrait du guide
    assert "Attendre" not in everything  # ni la réponse
    assert set(row) == {
        "ts",
        "day",
        "request_id",
        "question",
        "status",
        "refus_raison",
        "cited_chunk_ids",
        "latency_ms",
        "cost_usd",
        "prompt_version",
    }


# ------------------------------------------------------------------------------- CORS, démarrage
def test_cors_limited_to_configured_origin(make_client: Maker) -> None:
    client, _, _ = make_client()
    allowed = client.options(
        "/ask",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    other = client.options(
        "/ask",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in other.headers


def test_pipeline_built_once_by_lifespan(tmp_path: Path) -> None:
    calls = []

    def build() -> AppState:
        calls.append(1)
        log = RequestLog(tmp_path / "log.sqlite")
        return AppState(
            answerer=None,
            ready=lambda: ReadyStatus(False, False, None),
            log=log,
            settings=ApiSettings(1.0, [], log.path),
            budget=BudgetGuard(log, 1.0, 0.0),
        )

    with TestClient(create_app(build=build)) as client:
        for _ in range(3):
            client.get("/ready")
            client.post("/ask", json={"question": "Quelle procédure ?"})
    assert calls == [1]


class FakeCursor:
    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None: ...

    def execute(self, sql: str, params: object = None) -> None:
        self.sql = sql

    def fetchone(self) -> tuple[int]:
        return (1,)


class FakeConnection:
    def __init__(self) -> None:
        self.read_only = False
        self.closed = False

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def cursor(self) -> FakeCursor:
        return FakeCursor()


def test_build_state_opens_one_connection_per_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """La base n'est pas partagée : rien n'est ouvert au démarrage, chaque /ready ouvre sa
    propre connexion en lecture seule et la ferme. Aucun secret réel n'est lu."""
    import maintenance_rag.ingestion as ingestion

    class FakeSecrets:
        supabase_db_url = SecretStr("postgresql://fausse-url")
        openai_api_key = SecretStr("fausse-cle")

    opened: list[FakeConnection] = []

    def fake_connect(*args: object, **kwargs: object) -> FakeConnection:
        conn = FakeConnection()
        opened.append(conn)
        return conn

    monkeypatch.setattr(ingestion, "Settings", FakeSecrets)
    monkeypatch.setattr(dependencies.psycopg, "connect", fake_connect)
    monkeypatch.setattr(
        dependencies.ApiSettings,
        "from_config",
        classmethod(lambda cls: cls(1.0, [], tmp_path / "log.sqlite")),
    )
    state = dependencies.build_state()
    assert state.answerer is not None
    assert opened == []  # aucune connexion ouverte au démarrage
    assert state.ready().ready and state.ready().ready
    assert len(opened) == 2
    assert all(conn.read_only and conn.closed for conn in opened)
    assert state.budget.reservation_usd > 0  # coût maximal d'une question réservé


def test_api_does_not_use_cached_embedder() -> None:
    assert "CachedEmbedder(" not in inspect.getsource(dependencies)


def test_api_settings_from_config() -> None:
    settings = ApiSettings.from_config()
    assert settings.daily_budget_usd == 1.0
    assert settings.cors_origins == ["http://localhost:5173"]
    assert settings.log_path.name == "api_log.sqlite"
