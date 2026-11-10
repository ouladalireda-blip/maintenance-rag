"""Tests de la génération : prompt versionné, présentation des passages, schéma strict, analyse
de la sortie (aucun réseau : FakeLLM)."""

import json
from dataclasses import replace

from maintenance_rag.generation import (
    PROMPT_VERSION,
    SCHEMA,
    SYSTEM_PROMPT,
    build_user_message,
    generate,
    label_passages,
    parse,
    printed_pages,
)
from maintenance_rag.llm import FakeLLM, LLMReply, LLMSettings
from maintenance_rag.retrieval import SearchResult
from maintenance_rag.schemas import LLMAnswer

BASE = SearchResult(
    chunk_id=10,
    document_slug="inrs-ed6109",
    chunk_index=3,
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
    content="Texte du passage.",
)
GOOD_JSON = json.dumps(
    {
        "info_presente": True,
        "reponse": "Réponse.",
        "affirmations": [
            {"texte": "Affirmation.", "extrait_exact": "Texte du passage.", "passage_id": "P1"}
        ],
    }
)


def _reply(text: str, status: str = "completed") -> LLMReply:
    return LLMReply(text=text, input_tokens=10, output_tokens=5, status=status, model="fake")


def test_prompt_is_versioned_and_states_the_rules() -> None:
    assert PROMPT_VERSION.startswith("generation-v")
    for rule in (
        "UNIQUEMENT à partir des passages",
        "connaissances générales",
        "COPIÉ MOT POUR MOT",
        "au moins 20 caractères",
        "passage_id",
        "doit figurer dans son extrait_exact",
        "info_presente = false",
    ):
        assert rule in SYSTEM_PROMPT


def test_passages_get_short_ids_title_and_printed_page() -> None:
    passages = [BASE, replace(BASE, chunk_id=11, printed_page_start=None, printed_page_end=None)]
    labelled = label_passages(passages)
    assert list(labelled) == ["P1", "P2"]
    message = build_user_message("Quelle procédure ?", labelled)
    assert message.startswith("Question : Quelle procédure ?")
    assert "[P1] Section : 5.1. Consignation électrique — page 14\nTexte du passage." in message
    assert "[P2] Section : 5.1. Consignation électrique — page non numérotée" in message


def test_printed_pages() -> None:
    assert printed_pages(BASE) == "page 14"
    assert printed_pages(replace(BASE, printed_page_end=15)) == "pages 14-15"


def test_schema_is_strict() -> None:
    def objects(node: object) -> list[dict[str, object]]:
        found: list[dict[str, object]] = []
        if isinstance(node, dict):
            if node.get("type") == "object":
                found.append(node)
            for value in node.values():
                found += objects(value)
        elif isinstance(node, list):
            for value in node:
                found += objects(value)
        return found

    found = objects(SCHEMA)
    assert len(found) == 2  # la réponse et une affirmation
    for node in found:
        assert node["additionalProperties"] is False
        assert sorted(node["required"]) == sorted(node["properties"])  # type: ignore[call-overload]
    top = SCHEMA["properties"]
    assert set(top) == {"info_presente", "reponse", "affirmations"}  # type: ignore[arg-type]


def test_parse_valid_output() -> None:
    output, error = parse(_reply(GOOD_JSON))
    assert error is None
    assert output == LLMAnswer.model_validate_json(GOOD_JSON)


def test_parse_invalid_or_incomplete_output() -> None:
    assert parse(_reply("pas du JSON"))[0] is None
    extra = json.dumps({**json.loads(GOOD_JSON), "en_plus": 1})
    assert parse(_reply(extra))[0] is None  # champ en plus refusé
    missing = json.dumps({"info_presente": True, "reponse": "x"})
    assert parse(_reply(missing))[0] is None
    output, error = parse(_reply(GOOD_JSON, status="incomplete"))
    assert output is None
    assert error == "sortie incomplète (statut incomplete)"


def test_generate_sends_prompt_and_passages() -> None:
    llm = FakeLLM(lambda system, user: GOOD_JSON)
    generation = generate("Quelle procédure ?", [BASE], llm)
    assert generation.output is not None and generation.error is None
    assert generation.passages == {"P1": BASE}
    assert generation.prompt_version == PROMPT_VERSION
    ((system, user),) = llm.calls
    assert system == SYSTEM_PROMPT
    assert "[P1]" in user and "Texte du passage." in user


def test_cost() -> None:
    settings = LLMSettings("openai", "m", 0.0, 2000, 0.25, 2.0)
    assert settings.cost(1_000_000, 500_000) == 1.25
    assert replace(settings, price_per_1m_output=None).cost(1, 1) is None
