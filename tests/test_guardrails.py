"""Tests du garde-fou 2 : vérification déterministe des affirmations citées (aucun LLM)."""

from dataclasses import dataclass

import pytest

from maintenance_rag.guardrails import (
    Reason,
    Status,
    Verdict,
    describe,
    numbers,
    refused_reasons,
    verify,
)
from maintenance_rag.schemas import Affirmation, LLMAnswer


@dataclass(frozen=True)
class P:
    content: str


PASSAGES = {
    "P1": P("Le délai d'attente avant intervention est de 5 minutes après la coupure."),
    "P2": P("La tension doit être vérifiée sur chaque conducteur, y compris le neutre."),
}
QUOTE_1 = "Le délai d'attente avant intervention est de 5 minutes"
QUOTE_2 = "La tension doit être vérifiée sur chaque conducteur"


def _answer(*affirmations: Affirmation, present: bool = True) -> LLMAnswer:
    return LLMAnswer(info_presente=present, reponse="Réponse.", affirmations=list(affirmations))


def _aff(texte: str, extrait: str, passage_id: str = "P1") -> Affirmation:
    return Affirmation(texte=texte, extrait_exact=extrait, passage_id=passage_id)


GOOD_1 = _aff("Attendre 5 minutes après la coupure.", QUOTE_1)
GOOD_2 = _aff("Vérifier la tension sur chaque conducteur.", QUOTE_2, "P2")


def test_answered_when_everything_is_verified() -> None:
    verdict = verify(_answer(GOOD_1, GOOD_2), PASSAGES)
    assert verdict == Verdict(Status.ANSWERED)
    assert verdict.all_verified
    assert describe(verdict) == ""


def test_quote_matching_uses_normalization() -> None:
    spaced = _aff(
        "Attendre 5 minutes.", "LE DÉLAI  d'attente avant inter-\nvention est de 5 minutes"
    )
    assert verify(_answer(spaced), PASSAGES).status is Status.ANSWERED


def test_refused_by_llm_when_information_absent() -> None:
    verdict = verify(_answer(present=False), PASSAGES)
    assert verdict.status is Status.REFUSED_BY_LLM
    assert verdict.all_verified


def test_refused_by_llm_keeps_track_of_unverified_related_information() -> None:
    wrong = _aff("Information liée.", "phrase qui ne figure dans aucun passage")
    verdict = verify(_answer(GOOD_2, wrong, present=False), PASSAGES)
    assert verdict.status is Status.REFUSED_BY_LLM  # le refus du LLM reste un refus du LLM
    assert not verdict.all_verified  # mais les informations liées ne seront pas montrées


def test_invalid_output() -> None:
    verdict = verify(None, PASSAGES, detail="sortie invalide (1 erreur(s) de schéma)")
    assert verdict.status is Status.REFUSED_BY_GUARDRAIL
    assert verdict.reason is Reason.INVALID_OUTPUT
    assert describe(verdict) == "sortie_invalide : sortie invalide (1 erreur(s) de schéma)"


def test_empty_affirmation_list() -> None:
    verdict = verify(_answer(), PASSAGES)
    assert verdict.status is Status.REFUSED_BY_GUARDRAIL
    assert verdict.reason is Reason.NO_AFFIRMATION


def test_unknown_passage() -> None:
    verdict = verify(_answer(_aff("Attendre 5 minutes.", QUOTE_1, "P9")), PASSAGES)
    assert verdict.status is Status.REFUSED_BY_GUARDRAIL
    assert verdict.reason is Reason.UNKNOWN_PASSAGE
    assert describe(verdict) == "passage_inconnu (affirmation 1, P9)"


def test_quote_too_short() -> None:
    verdict = verify(_answer(_aff("Attendre.", "5 minutes")), PASSAGES)
    assert verdict.reason is Reason.QUOTE_TOO_SHORT
    assert verdict.status is Status.REFUSED_BY_GUARDRAIL


def test_quote_not_found_or_in_another_passage() -> None:
    altered = _aff("Attendre 5 minutes.", "Le délai d'attente avant intervention est de 10 minutes")
    assert verify(_answer(altered), PASSAGES).reason is Reason.QUOTE_NOT_FOUND
    wrong_passage = _aff("Attendre 5 minutes.", QUOTE_1, "P2")  # extrait vrai, mauvais passage
    assert verify(_answer(wrong_passage), PASSAGES).reason is Reason.QUOTE_NOT_FOUND


def test_number_not_quoted() -> None:
    verdict = verify(_answer(_aff("Attendre 10 minutes.", QUOTE_1)), PASSAGES)
    assert verdict.status is Status.REFUSED_BY_GUARDRAIL
    assert verdict.reason is Reason.NUMBER_NOT_QUOTED
    # « 5 » est cité, mais pas « 15 » : la comparaison porte sur des nombres entiers, pas
    # sur des sous-chaînes.
    assert verify(_answer(_aff("Attendre 15 minutes.", QUOTE_1)), PASSAGES).reason is (
        Reason.NUMBER_NOT_QUOTED
    )


def test_one_false_affirmation_refuses_the_whole_answer() -> None:
    bad = _aff("Attendre 10 minutes.", QUOTE_1)
    verdict = verify(_answer(GOOD_1, GOOD_2, bad), PASSAGES)
    assert verdict.status is Status.REFUSED_BY_GUARDRAIL
    assert [(f.affirmation, f.reason) for f in verdict.failures] == [(3, Reason.NUMBER_NOT_QUOTED)]


def test_every_failure_is_reported() -> None:
    verdict = verify(
        _answer(_aff("x", QUOTE_1, "P7"), GOOD_2, _aff("Attendre 9 minutes.", QUOTE_1)), PASSAGES
    )
    assert verdict.reason is Reason.UNKNOWN_PASSAGE  # la première cause
    assert describe(verdict) == (
        "passage_inconnu (affirmation 1, P7) ; nombre_non_cite (affirmation 3, P1)"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5 minutes", ["5"]),
        ("article R. 4544-11, 1 000 V, 0,5 s", ["4544", "11", "1", "000", "0", "5"]),
        ("aucun chiffre", []),
        ("５ minutes", ["5"]),  # chiffre pleine chasse, ramené par NFKC
    ],
)
def test_numbers(text: str, expected: list[str]) -> None:
    assert numbers(text) == expected


def test_refused_reasons() -> None:
    verdicts = [
        Verdict(Status.ANSWERED),
        Verdict(Status.REFUSED_BY_LLM),
        Verdict(Status.REFUSED_BY_GUARDRAIL, Reason.QUOTE_NOT_FOUND),
        Verdict(Status.REFUSED_BY_GUARDRAIL, Reason.QUOTE_NOT_FOUND),
        Verdict(Status.REFUSED_BY_GUARDRAIL, Reason.NO_AFFIRMATION),
    ]
    assert refused_reasons(verdicts) == {"aucune_affirmation": 1, "extrait_introuvable": 2}


def _answer_with(reponse: str, *affirmations: Affirmation, present: bool = True) -> LLMAnswer:
    return LLMAnswer(info_presente=present, reponse=reponse, affirmations=list(affirmations))


def test_number_in_answer_present_in_a_verified_quote() -> None:
    # « 5 » est dans l'extrait vérifié de l'affirmation 1, même si la réponse ne le reprend
    # pas dans la même affirmation.
    output = _answer_with("Attendre 5 minutes, puis vérifier.", GOOD_2, GOOD_1)
    assert verify(output, PASSAGES).status is Status.ANSWERED


def test_number_in_answer_absent_from_quotes() -> None:
    verdict = verify(_answer_with("Attendre 10 minutes.", GOOD_1, GOOD_2), PASSAGES)
    assert verdict.status is Status.REFUSED_BY_GUARDRAIL
    assert verdict.reason is Reason.NUMBER_NOT_QUOTED_ANSWER
    assert describe(verdict) == "nombre_non_cite_reponse (réponse)"
    # « 15 » n'est pas « 5 » : comparaison de nombres entiers, pas de sous-chaînes
    assert verify(_answer_with("Attendre 15 minutes.", GOOD_1), PASSAGES).reason is (
        Reason.NUMBER_NOT_QUOTED_ANSWER
    )


def test_answer_without_number() -> None:
    assert verify(_answer_with("Vérifier la tension.", GOOD_2), PASSAGES).status is Status.ANSWERED


def test_number_in_answer_only_counts_verified_quotes() -> None:
    # Le « 5 » ne figure que dans un extrait introuvable : il ne compte pas.
    fake = _aff("Attendre 5 minutes.", "Le délai d'attente est de 5 minutes selon le guide")
    output = _answer_with("Attendre 5 minutes.", fake, present=False)
    verdict = verify(output, PASSAGES)
    assert verdict.status is Status.REFUSED_BY_LLM  # le refus du LLM reste un refus du LLM
    assert [f.reason for f in verdict.failures] == [
        Reason.QUOTE_NOT_FOUND,
        Reason.NUMBER_NOT_QUOTED_ANSWER,
    ]
    assert not verdict.all_verified  # la réponse du LLM ne sera pas montrée


def test_refused_by_llm_answer_with_unquoted_number_is_not_shown() -> None:
    output = _answer_with("Le guide ne cite pas l'article R. 4544-99.", present=False)
    verdict = verify(output, PASSAGES)
    assert verdict.status is Status.REFUSED_BY_LLM
    assert not verdict.all_verified


def test_affirmation_failure_takes_precedence_over_answer_failure() -> None:
    verdict = verify(_answer_with("Attendre 10 minutes.", _aff("x", QUOTE_1, "P9")), PASSAGES)
    assert verdict.reason is Reason.UNKNOWN_PASSAGE


def test_bullets_and_footnote_refs_in_passage_do_not_block_an_exact_quote() -> None:
    passages = {
        "P1": P(
            "Opérations : • séparation de la source ; • mise à la terre[^4] immédiatement après."
        )
    }
    quote = "Opérations : séparation de la source ; mise à la terre immédiatement après."
    output = _answer(_aff("Séparer puis mettre à la terre.", quote))
    assert verify(output, passages).status is Status.ANSWERED
    # Un mot changé échoue toujours.
    changed = _aff("x", "Opérations : séparation de la source ; mise à la terre juste après.")
    assert verify(_answer(changed), passages).reason is Reason.QUOTE_NOT_FOUND
