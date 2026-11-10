"""Tests du pipeline de bout en bout avec un FakeLLM et une recherche factice (aucun réseau)."""

import json
from collections.abc import Sequence
from dataclasses import replace

import pytest

from maintenance_rag.guardrails import Status
from maintenance_rag.llm import FakeLLM, LLMSettings
from maintenance_rag.pipeline import (
    NOT_FOUND_MESSAGE,
    REFUSAL_MESSAGE,
    Pipeline,
    PipelineSettings,
)
from maintenance_rag.retrieval import SearchResult

CONTENT = "Le délai d'attente avant intervention est de 5 minutes après la coupure."
QUOTE = "Le délai d'attente avant intervention est de 5 minutes"
BASE = SearchResult(
    chunk_id=100,
    document_slug="inrs-ed6109",
    chunk_index=0,
    section_title="5.1. Consignation électrique",
    page_start=16,
    page_end=16,
    printed_page_start=14,
    printed_page_end=14,
    semantic_rank=1,
    full_text_rank=None,
    identifier_rank=None,
    rrf_score=0.02,
    has_warning=False,
    content=CONTENT,
)
CANDIDATES = [
    replace(BASE, chunk_id=100 + i, content=CONTENT if i == 0 else "Autre.") for i in range(20)
]
SETTINGS = PipelineSettings(
    top_k=8,
    min_quote_chars=20,
    rerank_enabled=False,
    llm=LLMSettings("openai", "fake", 0.0, 2000, 1.0, 10.0),
)


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


def _pipeline(text: str, **settings: object) -> tuple[Pipeline, FakeLLM]:
    llm = FakeLLM(lambda system, user: text, input_tokens=2000, output_tokens=300)
    return Pipeline(lambda q: CANDIDATES, llm, replace(SETTINGS, **settings)), llm  # type: ignore[arg-type]


def test_answered_with_citations_and_costs() -> None:
    pipeline, llm = _pipeline(_json(True, ("Attendre 5 minutes.", QUOTE, "P1")))
    result = pipeline.answer("Combien de temps attendre ?")
    assert result.status is Status.ANSWERED
    assert result.reason == ""
    assert result.reponse == "Attendre."
    (citation,) = result.citations
    assert (citation.passage_id, citation.chunk_id, citation.section_title) == (
        "P1",
        100,
        "5.1. Consignation électrique",
    )
    assert (citation.printed_page_start, citation.printed_page_end) == (14, 14)
    assert result.passage_ids == [100 + i for i in range(8)]  # top_k = 8 passages fournis
    assert result.input_tokens == 2000 and result.output_tokens == 300
    assert result.cost_usd == pytest.approx((2000 * 1.0 + 300 * 10.0) / 1_000_000)
    assert set(result.latencies_ms) == {"retrieval", "generation", "verification", "total"}
    ((_, user),) = llm.calls
    assert "[P8]" in user and "[P9]" not in user
    assert "Le délai" not in repr(result)  # le texte du guide n'est pas dans repr


def test_refused_by_llm_shows_verified_related_information() -> None:
    pipeline, _ = _pipeline(
        _json(False, ("Information liée : 5 minutes.", QUOTE, "P1"), reponse="Absent du guide.")
    )
    result = pipeline.answer("Quelle est la valeur ?")
    assert result.status is Status.REFUSED_BY_LLM
    assert result.reponse == "Absent du guide."
    assert len(result.citations) == 1


def test_refused_by_llm_hides_unverified_related_information() -> None:
    pipeline, _ = _pipeline(
        _json(False, ("Information liée.", "phrase absente de tout passage", "P1"))
    )
    result = pipeline.answer("Quelle est la valeur ?")
    assert result.status is Status.REFUSED_BY_LLM
    assert result.reponse == NOT_FOUND_MESSAGE
    assert result.citations == []


def test_refused_by_guardrail_shows_standard_message() -> None:
    pipeline, _ = _pipeline(_json(True, ("Attendre 10 minutes.", QUOTE, "P1")))
    result = pipeline.answer("Combien de temps attendre ?")
    assert result.status is Status.REFUSED_BY_GUARDRAIL
    assert result.reason == "nombre_non_cite (affirmation 1, P1)"
    assert result.reponse == REFUSAL_MESSAGE
    assert result.citations == []


def test_invalid_output_is_a_guardrail_refusal() -> None:
    pipeline, _ = _pipeline("{pas du json")
    result = pipeline.answer("Question ?")
    assert result.status is Status.REFUSED_BY_GUARDRAIL
    assert result.reason.startswith("sortie_invalide")


def test_no_passage_means_no_llm_call() -> None:
    llm = FakeLLM(lambda system, user: "{}")
    result = Pipeline(lambda q: [], llm, SETTINGS).answer("Question ?")
    assert result.status is Status.REFUSED_BY_GUARDRAIL
    assert result.reason == "aucun_passage"
    assert llm.calls == []


class ExplodingReranker:
    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        raise AssertionError("le reranker ne doit pas être appelé")


def test_rerank_disabled_is_respected() -> None:
    llm = FakeLLM(lambda system, user: _json(True, ("Attendre 5 minutes.", QUOTE, "P1")))
    pipeline = Pipeline(lambda q: CANDIDATES, llm, SETTINGS, reranker=ExplodingReranker())
    result = pipeline.answer("Combien de temps attendre ?")
    assert result.status is Status.ANSWERED
    assert "rerank" not in result.latencies_ms


def test_rerank_enabled_requires_a_reranker() -> None:
    with pytest.raises(ValueError):
        Pipeline(
            lambda q: CANDIDATES, FakeLLM(lambda s, u: "{}"), replace(SETTINGS, rerank_enabled=True)
        )


def test_from_config_does_not_load_the_model_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import maintenance_rag.rerank as rerank_module

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("FlashRank ne doit pas être chargé")

    monkeypatch.setattr(rerank_module.FlashRankReranker, "from_config", forbidden)
    pipeline = Pipeline.from_config(object(), object(), FakeLLM(lambda s, u: "{}"), SETTINGS)
    assert isinstance(pipeline, Pipeline)


def test_settings_from_config() -> None:
    settings = PipelineSettings.from_config()
    assert settings.top_k == 8
    assert settings.min_quote_chars == 20
    assert settings.rerank_enabled is False
    assert settings.llm.temperature == 0.0
