"""Client LLM : couche mince sur le fournisseur configuré (température 0, sortie structurée).

- ``LLM`` : interface (Protocol) : un message système, un message utilisateur, un JSON schema
  strict → ``LLMReply`` (texte brut renvoyé, tokens consommés, statut).
- ``OpenAILLM`` : API Responses d'OpenAI, Structured Outputs en mode ``strict``. Aucune reprise
  automatique : une erreur réseau ou SSL remonte telle quelle (l'appelant s'arrête), la
  vérification TLS n'est jamais modifiée. ``store=False`` : rien n'est conservé côté OpenAI.
- ``FakeLLM`` : réponses écrites d'avance, pour les tests (aucun réseau).

Le texte renvoyé n'est PAS interprété ici : sa validation (Pydantic) est faite par
``generation``, pour qu'une sortie invalide devienne un refus du garde-fou 2, pas une exception.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from maintenance_rag.config import CONFIG_PATH


@dataclass(frozen=True)
class LLMReply:
    text: str  # sortie brute (JSON attendu)
    input_tokens: int
    output_tokens: int
    status: str  # « completed », « incomplete » (plafond de sortie atteint)…
    model: str


class LLM(Protocol):
    model: str

    def complete(
        self, system: str, user: str, schema: Mapping[str, object], schema_name: str
    ) -> LLMReply: ...


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float | None
    max_output_tokens: int
    price_per_1m_input: float | None
    price_per_1m_output: float | None

    def cost(self, input_tokens: int, output_tokens: int) -> float | None:
        """Coût en dollars, ``None`` si un prix n'est pas renseigné."""
        if self.price_per_1m_input is None or self.price_per_1m_output is None:
            return None
        return (
            input_tokens * self.price_per_1m_input + output_tokens * self.price_per_1m_output
        ) / 1_000_000


def load_llm_settings(config_path: Path | None = None) -> LLMSettings:
    """Section ``llm`` de config.yaml ; ``model`` vide est permis ici (contrôlé à l'usage)."""
    path = config_path or CONFIG_PATH
    llm = yaml.safe_load(path.read_text(encoding="utf-8"))["llm"]

    def price(key: str) -> float | None:
        return None if llm.get(key) is None else float(llm[key])

    return LLMSettings(
        provider=str(llm["provider"]),
        model=str(llm.get("model") or "").strip(),
        temperature=None if llm.get("temperature") is None else float(llm["temperature"]),
        max_output_tokens=int(llm["max_output_tokens"]),
        price_per_1m_input=price("price_per_1m_input"),
        price_per_1m_output=price("price_per_1m_output"),
    )


class OpenAILLM:
    """Responses API d'OpenAI avec JSON schema strict."""

    def __init__(self, settings: LLMSettings, api_key: str, client: Any | None = None) -> None:
        if settings.provider != "openai":
            raise ValueError(f"fournisseur LLM non pris en charge : {settings.provider!r}")
        if not settings.model:
            raise ValueError("llm.model est vide dans config.yaml : à renseigner")
        self.model = settings.model
        self._settings = settings
        if client is None:
            import openai

            client = openai.OpenAI(api_key=api_key, max_retries=0)
        self._client = client

    def complete(
        self, system: str, user: str, schema: Mapping[str, object], schema_name: str
    ) -> LLMReply:
        options: dict[str, Any] = {}
        if self._settings.temperature is not None:
            options["temperature"] = self._settings.temperature
        response = self._client.responses.create(
            model=self.model,
            instructions=system,
            input=user,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": dict(schema),
                    "strict": True,
                }
            },
            max_output_tokens=self._settings.max_output_tokens,
            store=False,
            **options,
        )
        usage = response.usage
        return LLMReply(
            text=response.output_text,
            input_tokens=int(usage.input_tokens) if usage else 0,
            output_tokens=int(usage.output_tokens) if usage else 0,
            status=str(response.status or "unknown"),
            model=str(response.model or self.model),
        )


class FakeLLM:
    """LLM de test : ``respond(system, user)`` donne le texte brut renvoyé. Garde les appels."""

    def __init__(
        self, respond: Callable[[str, str], str], input_tokens: int = 1000, output_tokens: int = 100
    ) -> None:
        self.model = "fake"
        self._respond = respond
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls: list[tuple[str, str]] = []

    def complete(
        self, system: str, user: str, schema: Mapping[str, object], schema_name: str
    ) -> LLMReply:
        self.calls.append((system, user))
        return LLMReply(
            text=self._respond(system, user),
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            status="completed",
            model=self.model,
        )
