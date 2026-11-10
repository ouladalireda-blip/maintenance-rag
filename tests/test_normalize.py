"""Tests de la normalisation (garde-fou 2) et du recollage des césures."""

import pytest

from maintenance_rag.normalize import (
    build_vocabulary,
    dehyphenate,
    normalize_for_match,
    quote_matches,
)

# --------------------------------------------------------------------------------------------
# normalize_for_match
# --------------------------------------------------------------------------------------------


def test_court_circuit_variants_are_equivalent() -> None:
    forms = {normalize_for_match(t) for t in ("Court-Circuit", "court circuit", "courtcircuit")}
    assert forms == {"courtcircuit"}


def test_typographic_apostrophe_equals_ascii() -> None:
    assert normalize_for_match("l’équipement") == normalize_for_match("l'équipement")


def test_non_breaking_spaces_equal_regular_space() -> None:
    reference = normalize_for_match("absence de tension :")
    for space in ("\u00a0", "\u202f", "\u2009"):
        assert normalize_for_match(f"absence{space}de tension{space}:") == reference


def test_guillemets_are_unified() -> None:
    assert normalize_for_match("«\u00a0Opérations\u00a0»") == normalize_for_match('"Opérations"')
    assert normalize_for_match("“Opérations”") == '"opérations"'


@pytest.mark.parametrize("dash", ["-", "\u2010", "\u2011", "\u2013", "\u2014"])
def test_every_dash_is_removed(dash: str) -> None:
    assert normalize_for_match(f"mise{dash}à{dash}la{dash}terre") == "miseàlaterre"


def test_soft_hyphen_and_layout_characters_are_removed() -> None:
    assert normalize_for_match("consi\xadgnation") == "consignation"
    assert normalize_for_match("•\u2009\x07séparation") == "•séparation"


def test_nfkc_expands_ligatures() -> None:
    assert normalize_for_match("ﬁltre") == "filtre"


def test_meaningful_differences_are_kept() -> None:
    assert normalize_for_match("400 V") != normalize_for_match("230 V")
    assert normalize_for_match("ERR-402") != normalize_for_match("ERR-403")


def test_empty_text() -> None:
    assert normalize_for_match("") == ""
    assert normalize_for_match(" \u00a0\xad- ") == ""


# --------------------------------------------------------------------------------------------
# dehyphenate / build_vocabulary
# --------------------------------------------------------------------------------------------


def test_dehyphenate_joins_known_word() -> None:
    assert dehyphenate("les équipe-ments de protection", {"équipements"}) == (
        "les équipements de protection"
    )


def test_dehyphenate_keeps_case() -> None:
    assert dehyphenate("Équipe-ments", {"équipements"}) == "Équipements"


def test_dehyphenate_keeps_real_compound() -> None:
    assert dehyphenate("mise en court-circuit", {"court", "circuit"}) == "mise en court-circuit"


def test_dehyphenate_never_touches_identifiers() -> None:
    assert dehyphenate("code ERR-402", {"err402", "err"}) == "code ERR-402"


def test_dehyphenate_handles_chains() -> None:
    assert dehyphenate("obli-gatoire et ci-dessus", {"obligatoire"}) == "obligatoire et ci-dessus"


def test_dehyphenate_empty_text() -> None:
    assert dehyphenate("", {"équipements"}) == ""


def test_build_vocabulary() -> None:
    vocabulary = build_vocabulary(
        ["Les équipements de travail.", "mise en court-circuit", "la consi\xadgnation"]
    )
    assert {"équipements", "travail", "consignation", "court", "circuit"} <= vocabulary
    assert "courtcircuit" not in vocabulary
    assert build_vocabulary([]) == set()


def test_vocabulary_from_corpus_drives_dehyphenation() -> None:
    vocabulary = build_vocabulary(["Porter des équipements adaptés.", "un court-circuit"])
    text = "connaître les équipe-ments ; mise en court-circuit"
    assert dehyphenate(text, vocabulary) == "connaître les équipements ; mise en court-circuit"


# --------------------------------------------------------------------------------------------
# quote_matches
# --------------------------------------------------------------------------------------------


def test_quote_with_control_character_inside_word_matches() -> None:
    assert quote_matches(
        "la purge intermédiaire ouverte", "Vanne fermée, la purge intermédiaire ouverte."
    )


def test_really_different_word_does_not_match() -> None:
    source = "La séparation de l'équipement est réalisée en premier."
    assert quote_matches("La séparation de l'équipement est réalisée", source)
    assert not quote_matches("La condamnation de l'équipement est réalisée", source)


def test_quote_shorter_than_min_chars_raises() -> None:
    with pytest.raises(ValueError):
        quote_matches("la séparation", "la séparation de l'équipement")  # 12 caractères normalisés
