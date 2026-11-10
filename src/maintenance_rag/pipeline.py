"""Orchestration de bout en bout : question → recherche hybride → (reranking, si activé) →
top_k passages → génération structurée (LLM) → garde-fou 2 → réponse vérifiée ou refus.

- Pas de garde-fou 1 avant génération : suspendu (docs/adr/0002-retrait-reranker-flashrank.md).
  Le refus vient du LLM (``info_presente`` faux) ou du garde-fou 2 (``guardrails.verify``).
- ``rerank.enabled: false`` (config.yaml) : le modèle de reranking n'est ni importé ni chargé ;
  les passages gardent l'ordre de la recherche retenue.
- ``Pipeline.answer(question)`` → ``PipelineResult`` : statut, réponse, citations (passage,
  section, page imprimée), ids des passages fournis, latences par étape, tokens et coût.

Réponse montrée selon le statut :

- ``ANSWERED`` : la réponse du LLM et toutes ses affirmations citées (toutes vérifiées) ;
- ``REFUSED_BY_LLM`` : la réponse du LLM (« l'information n'est pas dans le guide ») et les
  informations liées SI toutes sont vérifiées ; sinon le message standard, sans citation ;
- ``REFUSED_BY_GUARDRAIL`` : le message standard, sans citation.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from maintenance_rag.config import CONFIG_PATH
from maintenance_rag.generation import PROMPT_VERSION, generate
from maintenance_rag.guardrails import Reason, Status, Verdict, describe, verify
from maintenance_rag.llm import LLM, LLMSettings, load_llm_settings
from maintenance_rag.rerank import Reranker, rerank
from maintenance_rag.retrieval import SearchResult
from maintenance_rag.schemas import LLMAnswer

NOT_FOUND_MESSAGE = "Les passages du guide retrouvés ne contiennent pas cette information."
REFUSAL_MESSAGE = (
    "Je ne peux pas donner de réponse vérifiable à partir du guide pour cette question."
)


@dataclass(frozen=True)
class PipelineSettings:
    top_k: int
    min_quote_chars: int
    rerank_enabled: bool
    llm: LLMSettings

    @classmethod
    def from_config(cls, config_path: Path = CONFIG_PATH) -> "PipelineSettings":
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        settings = cls(
            top_k=int(config["generation"]["top_k"]),
            min_quote_chars=int(config["guardrails"]["min_quote_chars"]),
            rerank_enabled=bool(config["rerank"]["enabled"]),
            llm=load_llm_settings(config_path),
        )
        if settings.top_k < 1:
            raise ValueError("generation.top_k doit valoir au moins 1")
        return settings


@dataclass(frozen=True)
class Citation:
    passage_id: str  # id court montré au LLM (P1…)
    chunk_id: int
    document_slug: str
    section_title: str
    printed_page_start: int | None
    printed_page_end: int | None
    texte: str = field(repr=False)
    extrait_exact: str = field(repr=False)


@dataclass(frozen=True)
class PipelineResult:
    status: Status
    reason: str  # vide si ANSWERED ; codes et ids seulement, jamais de texte du guide
    reponse: str = field(repr=False)
    citations: list[Citation]
    passage_ids: list[int]  # chunk_id des passages fournis au LLM, dans l'ordre P1…Pn
    latencies_ms: dict[str, float]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    model: str
    prompt_version: str
    verdict: Verdict
    llm_output: LLMAnswer | None = field(default=None, repr=False)


class Pipeline:
    def __init__(
        self,
        retrieve: Callable[[str], Sequence[SearchResult]],
        llm: LLM,
        settings: PipelineSettings,
        reranker: Reranker | None = None,
    ) -> None:
        if settings.rerank_enabled and reranker is None:
            raise ValueError("rerank.enabled vaut true mais aucun reranker n'est fourni")
        self._retrieve = retrieve
        self._llm = llm
        self._settings = settings
        self._reranker = reranker if settings.rerank_enabled else None

    @classmethod
    def from_config(
        cls, embedder: Any, conn: Any, llm: LLM, settings: PipelineSettings | None = None
    ) -> "Pipeline":
        """Recherche retenue de config.yaml ; reranker chargé seulement si rerank.enabled."""
        from maintenance_rag.retrieval import search

        settings = settings or PipelineSettings.from_config()
        reranker: Reranker | None = None
        if settings.rerank_enabled:
            from maintenance_rag.rerank import FlashRankReranker

            reranker = FlashRankReranker.from_config()
        return cls(lambda question: search(question, embedder, conn), llm, settings, reranker)

    def answer(self, question: str) -> PipelineResult:
        latencies: dict[str, float] = {}
        start = clock = time.perf_counter()

        def lap(step: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            latencies[step] = (now - clock) * 1000
            clock = now

        candidates = list(self._retrieve(question))
        lap("retrieval")
        if self._reranker is not None:
            candidates = [r.result for r in rerank(question, candidates, self._reranker)]
            lap("rerank")
        passages = candidates[: self._settings.top_k]

        if not passages:
            verdict = Verdict(Status.REFUSED_BY_GUARDRAIL, Reason.NO_PASSAGE)
            latencies["total"] = (time.perf_counter() - start) * 1000
            return PipelineResult(
                status=verdict.status,
                reason=describe(verdict),
                reponse=REFUSAL_MESSAGE,
                citations=[],
                passage_ids=[],
                latencies_ms=latencies,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                model=self._llm.model,
                prompt_version=PROMPT_VERSION,
                verdict=verdict,
            )

        generation = generate(question, passages, self._llm)
        lap("generation")
        verdict = verify(
            generation.output,
            generation.passages,
            self._settings.min_quote_chars,
            detail=generation.error,
        )
        lap("verification")
        latencies["total"] = (time.perf_counter() - start) * 1000

        output = generation.output
        show_citations = output is not None and (
            verdict.status is Status.ANSWERED
            or (verdict.status is Status.REFUSED_BY_LLM and verdict.all_verified)
        )
        if verdict.status is Status.REFUSED_BY_GUARDRAIL or output is None:
            reponse = REFUSAL_MESSAGE
        elif verdict.status is Status.REFUSED_BY_LLM and not verdict.all_verified:
            reponse = NOT_FOUND_MESSAGE
        else:
            reponse = output.reponse
        citations = []
        if show_citations and output is not None:
            for affirmation in output.affirmations:
                passage = generation.passages[affirmation.passage_id]
                citations.append(
                    Citation(
                        passage_id=affirmation.passage_id,
                        chunk_id=passage.chunk_id,
                        document_slug=passage.document_slug,
                        section_title=passage.section_title,
                        printed_page_start=passage.printed_page_start,
                        printed_page_end=passage.printed_page_end,
                        texte=affirmation.texte,
                        extrait_exact=affirmation.extrait_exact,
                    )
                )
        reply = generation.reply
        return PipelineResult(
            status=verdict.status,
            reason=describe(verdict),
            reponse=reponse,
            citations=citations,
            passage_ids=[passage.chunk_id for passage in passages],
            latencies_ms=latencies,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            cost_usd=self._settings.llm.cost(reply.input_tokens, reply.output_tokens),
            model=reply.model,
            prompt_version=generation.prompt_version,
            verdict=verdict,
            llm_output=output,
        )
