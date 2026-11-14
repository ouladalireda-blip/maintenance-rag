"""Client HTTP de l'API maintenance-rag, utilisé par l'interface Streamlit.

- URL de l'API : variable d'environnement ``API_URL`` (par défaut http://localhost:8000).
- Délai d'attente : ``ui.timeout_s`` de config/config.yaml (60 s par défaut).
- Toute erreur devient une ``ApiError`` portant un message clair pour le technicien :
    * 503 → service momentanément indisponible ;
    * 429 → limite d'utilisation du jour atteinte ;
    * 422 → question trop courte ou trop longue ;
    * API injoignable (connexion refusée, DNS, délai dépassé…) → impossible de joindre le service.
- Aucun accès à la base ni au LLM : l'interface ne connaît que l'API.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_S = 60.0

# Avertissement de l'API (identique, vérifié par les tests), affiché avant la 1re question.
AVERTISSEMENT = (
    "Outil d'aide à la lecture du guide INRS ED 6109. La procédure de consignation de votre "
    "site et le chargé de consignation font foi."
)

MESSAGE_503 = "Service momentanément indisponible, réessayez dans un instant."
MESSAGE_429 = "Limite d'utilisation du jour atteinte."
MESSAGE_422 = "Question trop courte ou trop longue (3 à 500 caractères)."
MESSAGE_UNREACHABLE = "Impossible de joindre le service."
MESSAGE_FEEDBACK_404 = "Cette réponse n'est plus connue du service : avis non enregistré."
MESSAGE_INVALID = "Réponse inattendue du service."

Status = Literal["ANSWERED", "REFUSED_BY_LLM", "REFUSED_BY_GUARDRAIL"]
STATUSES: tuple[str, ...] = ("ANSWERED", "REFUSED_BY_LLM", "REFUSED_BY_GUARDRAIL")


class ApiError(Exception):
    """Erreur présentable telle quelle au technicien (``message``)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class Citation:
    texte: str
    extrait_exact: str
    page_imprimee: str | None
    section: str
    document: str


@dataclass(frozen=True)
class Answer:
    request_id: str
    status: Status
    reponse: str
    citations: list[Citation]
    refus_raison: str | None
    avertissement: str
    prompt_version: str
    latence_ms: float
    cout_usd: float | None


def error_message(status_code: int) -> str:
    """Message clair pour un code HTTP d'erreur de l'API."""
    messages = {503: MESSAGE_503, 429: MESSAGE_429, 422: MESSAGE_422}
    return messages.get(status_code, f"Erreur inattendue du service (HTTP {status_code}).")


def parse_answer(data: Any) -> Answer:
    """Réponse JSON de /ask → ``Answer`` ; ``ApiError`` si elle ne respecte pas le contrat."""
    try:
        status = data["status"]
        if status not in STATUSES:
            raise ValueError(status)
        cost = data["cout_usd"]
        return Answer(
            request_id=str(data["request_id"]),
            status=status,
            reponse=str(data["reponse"]),
            citations=[
                Citation(
                    texte=str(c["texte"]),
                    extrait_exact=str(c["extrait_exact"]),
                    page_imprimee=None if c["page_imprimee"] is None else str(c["page_imprimee"]),
                    section=str(c["section"]),
                    document=str(c["document"]),
                )
                for c in data["citations"]
            ],
            refus_raison=data["refus_raison"],
            avertissement=str(data["avertissement"]),
            prompt_version=str(data["prompt_version"]),
            latence_ms=float(data["latence_ms"]),
            cout_usd=None if cost is None else float(cost),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ApiError(MESSAGE_INVALID) from error


def configured_timeout(config_path: Path = ROOT / "config" / "config.yaml") -> float:
    try:
        ui = yaml.safe_load(config_path.read_text(encoding="utf-8")).get("ui") or {}
        return float(ui.get("timeout_s", DEFAULT_TIMEOUT_S))
    except (OSError, ValueError, AttributeError):
        return DEFAULT_TIMEOUT_S


class ApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("API_URL") or DEFAULT_API_URL).rstrip("/")
        self.timeout_s = configured_timeout() if timeout_s is None else timeout_s
        self._transport = transport

    def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        try:
            with httpx.Client(
                base_url=self.base_url, timeout=self.timeout_s, transport=self._transport
            ) as client:
                return client.post(path, json=payload)
        except httpx.TransportError as error:  # connexion refusée, DNS, délai dépassé…
            raise ApiError(MESSAGE_UNREACHABLE) from error

    def ask(self, question: str) -> Answer:
        response = self._post("/ask", {"question": question})
        if response.status_code != 200:
            raise ApiError(error_message(response.status_code), response.status_code)
        try:
            data = response.json()
        except ValueError as error:
            raise ApiError(MESSAGE_INVALID, response.status_code) from error
        return parse_answer(data)

    def feedback(self, request_id: str, utile: bool) -> None:
        response = self._post("/feedback", {"request_id": request_id, "utile": utile})
        if response.status_code == 204:
            return
        if response.status_code == 404:
            raise ApiError(MESSAGE_FEEDBACK_404, 404)
        raise ApiError(error_message(response.status_code), response.status_code)
