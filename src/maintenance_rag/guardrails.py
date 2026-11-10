"""Garde-fou 2 : vérification déterministe, sans aucun appel au LLM, de la sortie structurée.

Le garde-fou 1 (seuil τ sur un score avant génération) est suspendu : ni le reranker ni la
similarité ne séparent les pièges (docs/adr/0002-retrait-reranker-flashrank.md). Le refus
repose sur le LLM (``info_presente``) et sur cette vérification.

``verify(output, passages)`` contrôle chaque affirmation :

1. son ``passage_id`` fait partie des passages fournis ;
2. son ``extrait_exact`` figure mot pour mot dans le texte de ce passage
   (``quote_matches``, au moins ``min_chars`` caractères après normalisation) ;
3. chaque nombre (suite de chiffres) de son ``texte`` est aussi une suite de chiffres de son
   ``extrait_exact``.

Puis la ``reponse`` elle-même : chacun de ses nombres doit figurer dans au moins un
``extrait_exact`` VÉRIFIÉ de la même sortie (raison ``nombre_non_cite_reponse``). Avec
info_presente vrai, un échec donne ``REFUSED_BY_GUARDRAIL``. Avec info_presente faux, le statut
reste ``REFUSED_BY_LLM`` mais l'échec est enregistré (``all_verified`` faux) : la réponse du LLM
n'est alors pas montrée.

Statuts :

- ``ANSWERED`` : info_presente vrai, au moins une affirmation, toutes vérifiées ;
- ``REFUSED_BY_LLM`` : info_presente faux (le LLM dit que l'information est absente) ;
- ``REFUSED_BY_GUARDRAIL`` : sortie absente ou invalide, aucune affirmation, ou une
  vérification qui échoue. Règle stricte : UNE affirmation fausse refuse toute la réponse.

Les raisons ne contiennent que des codes, des numéros d'affirmation et des ids de passage,
jamais de texte du guide.
"""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from maintenance_rag.normalize import quote_matches
from maintenance_rag.schemas import Affirmation, LLMAnswer


class Status(StrEnum):
    ANSWERED = "ANSWERED"
    REFUSED_BY_LLM = "REFUSED_BY_LLM"
    REFUSED_BY_GUARDRAIL = "REFUSED_BY_GUARDRAIL"


class Reason(StrEnum):
    NO_PASSAGE = "aucun_passage"  # la recherche n'a rien renvoyé : le LLM n'est pas appelé
    INVALID_OUTPUT = "sortie_invalide"
    NO_AFFIRMATION = "aucune_affirmation"
    UNKNOWN_PASSAGE = "passage_inconnu"
    QUOTE_TOO_SHORT = "extrait_trop_court"
    QUOTE_NOT_FOUND = "extrait_introuvable"
    NUMBER_NOT_QUOTED = "nombre_non_cite"
    NUMBER_NOT_QUOTED_ANSWER = "nombre_non_cite_reponse"


ANSWER = 0  # numéro de « l'affirmation » quand l'échec porte sur le champ reponse


class Passage(Protocol):
    @property
    def content(self) -> str: ...


@dataclass(frozen=True)
class Failure:
    affirmation: int  # numéro (dès 1) ; ANSWER (0) pour le champ reponse
    reason: Reason
    passage_id: str  # vide pour le champ reponse


@dataclass(frozen=True)
class Verdict:
    status: Status
    reason: Reason | None = None  # première cause du refus par le garde-fou
    failures: list[Failure] = field(default_factory=list)  # toutes les affirmations fautives
    detail: str | None = None  # ex. raison d'une sortie invalide (sans texte du guide)

    @property
    def all_verified(self) -> bool:
        return not self.failures


_DIGITS = re.compile(r"\d+")


def numbers(text: str) -> list[str]:
    """Suites de chiffres du texte (après NFKC : chiffres en exposant, pleine chasse…)."""
    return _DIGITS.findall(unicodedata.normalize("NFKC", text))


def check(
    number: int, affirmation: Affirmation, passages: Mapping[str, Passage], min_chars: int
) -> Failure | None:
    """Première vérification qui échoue pour cette affirmation, ``None`` si tout est vérifié."""
    passage = passages.get(affirmation.passage_id)
    if passage is None:
        return Failure(number, Reason.UNKNOWN_PASSAGE, affirmation.passage_id)
    try:
        found = quote_matches(affirmation.extrait_exact, passage.content, min_chars)
    except ValueError:
        return Failure(number, Reason.QUOTE_TOO_SHORT, affirmation.passage_id)
    if not found:
        return Failure(number, Reason.QUOTE_NOT_FOUND, affirmation.passage_id)
    quoted = set(numbers(affirmation.extrait_exact))
    if any(value not in quoted for value in numbers(affirmation.texte)):
        return Failure(number, Reason.NUMBER_NOT_QUOTED, affirmation.passage_id)
    return None


def verify(
    output: LLMAnswer | None,
    passages: Mapping[str, Passage],
    min_chars: int = 20,
    detail: str | None = None,
) -> Verdict:
    """Statut de la sortie du LLM vis-à-vis des passages fournis (ids courts → passages)."""
    if output is None:
        return Verdict(Status.REFUSED_BY_GUARDRAIL, Reason.INVALID_OUTPUT, detail=detail)
    failures: list[Failure] = []
    verified_numbers: set[str] = set()  # nombres des extraits VÉRIFIÉS
    for number, affirmation in enumerate(output.affirmations, start=1):
        failure = check(number, affirmation, passages, min_chars)
        if failure is None:
            verified_numbers.update(numbers(affirmation.extrait_exact))
        else:
            failures.append(failure)
    answer_failures = (
        [Failure(ANSWER, Reason.NUMBER_NOT_QUOTED_ANSWER, "")]
        if any(value not in verified_numbers for value in numbers(output.reponse))
        else []
    )
    if not output.info_presente:
        # Refus du LLM ; affirmations « liées » et reponse vérifiées pour savoir si on les montre.
        return Verdict(Status.REFUSED_BY_LLM, failures=failures + answer_failures)
    if not output.affirmations:
        return Verdict(Status.REFUSED_BY_GUARDRAIL, Reason.NO_AFFIRMATION)
    if failures:
        return Verdict(Status.REFUSED_BY_GUARDRAIL, failures[0].reason, failures=failures)
    if answer_failures:
        return Verdict(Status.REFUSED_BY_GUARDRAIL, answer_failures[0].reason, answer_failures)
    return Verdict(Status.ANSWERED)


def describe(verdict: Verdict) -> str:
    """Raison lisible, sans texte du guide : « extrait_introuvable (affirmation 2, P3) »."""
    if verdict.reason is None:
        return ""
    parts = [
        f"{f.reason.value} (réponse)"
        if f.affirmation == ANSWER
        else f"{f.reason.value} (affirmation {f.affirmation}, {f.passage_id})"
        for f in verdict.failures
    ]
    if not parts:
        parts = [verdict.reason.value + (f" : {verdict.detail}" if verdict.detail else "")]
    return " ; ".join(parts)


def refused_reasons(verdicts: Sequence[Verdict]) -> dict[str, int]:
    """Nombre de refus du garde-fou par raison principale."""
    counts: dict[str, int] = {}
    for verdict in verdicts:
        if verdict.status is Status.REFUSED_BY_GUARDRAIL and verdict.reason is not None:
            counts[verdict.reason.value] = counts.get(verdict.reason.value, 0) + 1
    return dict(sorted(counts.items()))
