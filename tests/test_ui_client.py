"""Tests du client HTTP de l'interface (ui/client.py) avec un faux serveur HTTP
(``httpx.MockTransport``) : aucun réseau réel, aucune API lancée."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from ui.client import (
    AVERTISSEMENT,
    DEFAULT_API_URL,
    MESSAGE_422,
    MESSAGE_429,
    MESSAGE_503,
    MESSAGE_FEEDBACK_404,
    MESSAGE_INVALID,
    MESSAGE_UNREACHABLE,
    Answer,
    ApiClient,
    ApiError,
    Citation,
    error_message,
    parse_answer,
)

from maintenance_rag.api.schemas import AVERTISSEMENT as API_AVERTISSEMENT

ROOT = Path(__file__).resolve().parents[1]

ANSWER_JSON: dict[str, Any] = {
    "request_id": "a" * 32,
    "status": "ANSWERED",
    "reponse": "Réponse courte.",
    "citations": [
        {
            "texte": "Affirmation.",
            "extrait_exact": "Extrait vérifié mot pour mot.",
            "page_imprimee": "14-15",
            "section": "5.1. Consignation électrique",
            "document": "INRS ED 6109 — Consignations et déconsignations",
        }
    ],
    "refus_raison": None,
    "avertissement": API_AVERTISSEMENT,
    "prompt_version": "generation-v2-2026-11-11",
    "latence_ms": 1234.5,
    "cout_usd": 0.0005,
}


class FakeServer:
    """Faux serveur HTTP : répond selon le chemin et garde les requêtes reçues."""

    def __init__(self, routes: dict[str, httpx.Response]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.routes[request.url.path]


def _client(routes: dict[str, httpx.Response]) -> tuple[ApiClient, FakeServer]:
    server = FakeServer(routes)
    return ApiClient("http://api.test", timeout_s=5, transport=httpx.MockTransport(server)), server


def test_ask_parses_the_answer() -> None:
    client, server = _client({"/ask": httpx.Response(200, json=ANSWER_JSON)})
    answer = client.ask("Quelle procédure ?")
    assert answer == Answer(
        request_id="a" * 32,
        status="ANSWERED",
        reponse="Réponse courte.",
        citations=[
            Citation(
                texte="Affirmation.",
                extrait_exact="Extrait vérifié mot pour mot.",
                page_imprimee="14-15",
                section="5.1. Consignation électrique",
                document="INRS ED 6109 — Consignations et déconsignations",
            )
        ],
        refus_raison=None,
        avertissement=API_AVERTISSEMENT,
        prompt_version="generation-v2-2026-11-11",
        latence_ms=1234.5,
        cout_usd=0.0005,
    )
    (request,) = server.requests
    assert request.method == "POST" and str(request.url) == "http://api.test/ask"
    assert json.loads(request.content) == {"question": "Quelle procédure ?"}


@pytest.mark.parametrize("status", ["REFUSED_BY_LLM", "REFUSED_BY_GUARDRAIL"])
def test_ask_parses_refusals(status: str) -> None:
    body = {
        **ANSWER_JSON,
        "status": status,
        "citations": [],
        "refus_raison": None,
        "cout_usd": None,
    }
    client, _ = _client({"/ask": httpx.Response(200, json=body)})
    answer = client.ask("Quelle procédure ?")
    assert answer.status == status and answer.citations == [] and answer.cout_usd is None


@pytest.mark.parametrize(
    ("code", "message"),
    [(503, MESSAGE_503), (429, MESSAGE_429), (422, MESSAGE_422)],
)
def test_error_codes_are_translated(code: int, message: str) -> None:
    client, _ = _client({"/ask": httpx.Response(code, json={"detail": "message technique"})})
    with pytest.raises(ApiError) as caught:
        client.ask("Quelle procédure ?")
    assert caught.value.message == message
    assert caught.value.status_code == code
    assert "message technique" not in caught.value.message


def test_messages_are_the_expected_ones() -> None:
    assert error_message(503) == "Service momentanément indisponible, réessayez dans un instant."
    assert error_message(429) == "Limite d'utilisation du jour atteinte."
    assert error_message(422) == "Question trop courte ou trop longue (3 à 500 caractères)."
    assert MESSAGE_UNREACHABLE == "Impossible de joindre le service."
    assert error_message(500) == "Erreur inattendue du service (HTTP 500)."


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("connexion refusée"),
        httpx.ConnectTimeout("délai de connexion"),
        httpx.ReadTimeout("délai de lecture"),
    ],
    ids=["connexion", "delai-connexion", "delai-lecture"],
)
def test_unreachable_api(error: Exception) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise error

    client = ApiClient("http://api.test", timeout_s=5, transport=httpx.MockTransport(refuse))
    with pytest.raises(ApiError) as caught:
        client.ask("Quelle procédure ?")
    assert caught.value.message == MESSAGE_UNREACHABLE
    assert caught.value.status_code is None


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"pas du json"),
        httpx.Response(200, json={"status": "ANSWERED"}),  # champs manquants
        httpx.Response(200, json={**ANSWER_JSON, "status": "PEUT-ETRE"}),
    ],
    ids=["pas-json", "incomplet", "statut-inconnu"],
)
def test_unexpected_answer_is_an_api_error(response: httpx.Response) -> None:
    client, _ = _client({"/ask": response})
    with pytest.raises(ApiError) as caught:
        client.ask("Quelle procédure ?")
    assert caught.value.message == MESSAGE_INVALID


def test_parse_answer_rejects_bad_types() -> None:
    with pytest.raises(ApiError):
        parse_answer({**ANSWER_JSON, "latence_ms": "rapide"})
    with pytest.raises(ApiError):
        parse_answer(["pas", "un", "objet"])


def test_feedback() -> None:
    client, server = _client({"/feedback": httpx.Response(204)})
    client.feedback("a" * 32, True)
    (request,) = server.requests
    assert str(request.url) == "http://api.test/feedback"
    assert json.loads(request.content) == {"request_id": "a" * 32, "utile": True}


@pytest.mark.parametrize(
    ("code", "message"), [(404, MESSAGE_FEEDBACK_404), (503, MESSAGE_503), (422, MESSAGE_422)]
)
def test_feedback_errors(code: int, message: str) -> None:
    client, _ = _client({"/feedback": httpx.Response(code, json={"detail": "x"})})
    with pytest.raises(ApiError) as caught:
        client.feedback("a" * 32, False)
    assert caught.value.message == message


def test_api_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_URL", raising=False)
    assert ApiClient(timeout_s=1).base_url == DEFAULT_API_URL == "http://localhost:8000"
    monkeypatch.setenv("API_URL", "http://api.interne:9000/")
    assert ApiClient(timeout_s=1).base_url == "http://api.interne:9000"


def test_timeout_from_config() -> None:
    assert (
        ApiClient().timeout_s
        == yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))["ui"][
            "timeout_s"
        ]
    )


def test_warning_matches_the_api() -> None:
    assert AVERTISSEMENT == API_AVERTISSEMENT


def test_examples_come_only_from_the_dev_set() -> None:
    ui = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))["ui"]
    dev = yaml.safe_load((ROOT / "evals" / "questions_dev.yaml").read_text(encoding="utf-8"))
    categories = {q["id"]: q["category"] for q in dev["questions"]}
    assert all(qid.startswith("INRS-DEV-") and qid in categories for qid in ui["examples"])
    kinds = [categories[qid] for qid in ui["examples"]]
    assert kinds.count("valid") == 2 and len(kinds) - kinds.count("valid") == 2  # 2 pièges
    source = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    assert "questions_test" not in source and "INRS-TEST" not in source
