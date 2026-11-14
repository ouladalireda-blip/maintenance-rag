"""Schémas Pydantic des requêtes et réponses de l'API (contrat HTTP, distinct des modèles
internes du pipeline).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

AVERTISSEMENT = (
    "Outil d'aide à la lecture du guide INRS ED 6109. La procédure de consignation de votre "
    "site et le chargé de consignation font foi."
)

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question = Field(description="question du technicien (3 à 500 caractères)")


class CitationOut(BaseModel):
    texte: str = Field(description="affirmation, adossée à l'extrait")
    extrait_exact: str = Field(description="extrait du guide, vérifié mot pour mot")
    page_imprimee: str | None = Field(description="page imprimée (« 14 » ou « 14-15 »)")
    section: str
    document: str


class AskResponse(BaseModel):
    request_id: str
    status: Literal["ANSWERED", "REFUSED_BY_LLM", "REFUSED_BY_GUARDRAIL"]
    reponse: str
    citations: list[CitationOut]
    refus_raison: str | None = Field(description="code du refus du garde-fou, null sinon")
    avertissement: str = AVERTISSEMENT
    prompt_version: str
    latence_ms: float
    cout_usd: float | None


RequestId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: RequestId = Field(description="request_id renvoyé par /ask")
    utile: bool


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    database: bool
    llm_config: bool
    prompt_version: str | None


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
