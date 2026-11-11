"""Application FastAPI : routes (santé, disponibilité, question/réponse) ; point d'entrée
uvicorn ``maintenance_rag.api.main:app``.

- Le pipeline est créé UNE fois, au démarrage (lifespan), par ``build_state``.
- ``GET /health`` : vivant, sans aucun appel externe (HEALTHCHECK Docker).
- ``GET /ready`` : la base répond (``select 1`` en lecture seule) et la config LLM est
  chargée ; sinon 503. Ni secret ni URL de connexion dans la réponse.
- ``POST /ask`` : question → réponse vérifiée ou refus. L'API ne change rien au pipeline
  (recherche, prompt, garde-fou, normalisation) ; elle le présente :
    * la ``reponse`` est celle produite par le pipeline, y compris ses messages standards de
      refus ; les informations liées d'un refus ne sont présentes que si toutes sont vérifiées
      (comportement du pipeline) ;
    * ``avertissement`` toujours présent ;
    * erreur réseau, TLS, OpenAI ou base : 503, jamais présentée comme un refus ;
    * entrée invalide : 422 ; budget journalier épuisé : 429, sans appel au LLM.
- Chaque appel traité est journalisé (``api.logging``), sans réponse ni extrait du guide.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .dependencies import UPSTREAM_ERRORS, ApiSettings, AppState, build_state, get_state
from .schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    ErrorResponse,
    HealthResponse,
    ReadyResponse,
)

logger = logging.getLogger("maintenance_rag.api")

UPSTREAM_MESSAGE = (
    "Service momentanément indisponible (base de données ou modèle de langage injoignable). "
    "Ce n'est pas une réponse du guide : réessayez plus tard."
)
NOT_READY_MESSAGE = "Service non prêt : le pipeline n'a pas pu être initialisé."
BUDGET_MESSAGE = (
    "Budget journalier de l'assistant atteint : aucune nouvelle question n'est traitée "
    "aujourd'hui. Consultez directement le guide INRS ED 6109 ou le chargé de consignation."
)

State = Annotated[AppState, Depends(get_state)]


def _pages(start: int | None, end: int | None) -> str | None:
    if start is None:
        return None
    return str(start) if end in (None, start) else f"{start}-{end}"


def _error(code: int, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content=ErrorResponse(detail=message, request_id=request_id).model_dump(),
    )


def create_app(build: Callable[[], AppState] = build_state) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.rag = build()  # une seule fois, au démarrage
        yield

    app = FastAPI(
        title="maintenance-rag",
        description="Questions sur le guide INRS ED 6109, réponses citées et vérifiées.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ApiSettings.from_config().cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}})
    def ready(state: State, response: Response) -> ReadyResponse:
        check = state.ready()
        if not check.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(
            status="ready" if check.ready else "not_ready",
            database=check.database,
            llm_config=check.llm_config,
            prompt_version=check.prompt_version,
        )

    @app.post(
        "/ask",
        response_model=AskResponse,
        responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def ask(body: AskRequest, state: State) -> AskResponse | JSONResponse:
        request_id = uuid.uuid4().hex
        if state.answerer is None:
            return _error(status.HTTP_503_SERVICE_UNAVAILABLE, NOT_READY_MESSAGE, request_id)
        if not state.budget.reserve():
            return _error(status.HTTP_429_TOO_MANY_REQUESTS, BUDGET_MESSAGE, request_id)

        entry: dict[str, Any] = {
            "request_id": request_id,
            "question": body.question,
            "status": "UPSTREAM_ERROR",
            "refus_raison": None,
            "cited_chunk_ids": [],
            "cost_usd": None,
            "prompt_version": None,
        }
        start = time.perf_counter()
        try:
            result = state.answerer.answer(body.question)
        except UPSTREAM_ERRORS as error:
            logger.warning("erreur en amont %s (request_id %s)", type(error).__name__, request_id)
            entry["latency_ms"] = (time.perf_counter() - start) * 1000
            state.budget.settle(lambda: state.log.record(**entry))
            return _error(status.HTTP_503_SERVICE_UNAVAILABLE, UPSTREAM_MESSAGE, request_id)
        except BaseException:
            entry["latency_ms"] = (time.perf_counter() - start) * 1000
            state.budget.settle(lambda: state.log.record(**entry))
            raise
        latency = (time.perf_counter() - start) * 1000

        reason = result.verdict.reason.value if result.verdict.reason else None
        entry.update(
            status=result.status.value,
            refus_raison=reason,
            cited_chunk_ids=[c.chunk_id for c in result.citations],
            latency_ms=latency,
            cost_usd=result.cost_usd,
            prompt_version=result.prompt_version,
        )
        state.budget.settle(lambda: state.log.record(**entry))
        return AskResponse(
            request_id=request_id,
            status=result.status.value,
            reponse=result.reponse,  # réponse du pipeline, messages standards de refus compris
            citations=[
                CitationOut(
                    texte=c.texte,
                    extrait_exact=c.extrait_exact,
                    page_imprimee=_pages(c.printed_page_start, c.printed_page_end),
                    section=c.section_title,
                    document=state.documents.get(c.document_slug, c.document_slug),
                )
                for c in result.citations
            ],
            refus_raison=reason,
            prompt_version=result.prompt_version,
            latence_ms=round(latency, 1),
            cout_usd=result.cost_usd,
        )

    return app


app = create_app()
