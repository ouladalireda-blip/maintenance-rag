"""Normalisation du texte : forme de comparaison du garde-fou 2 et recollage des césures.

Deux usages distincts, à ne pas confondre :

- ``normalize_for_match`` produit une forme de COMPARAISON, illisible et jamais affichée ;
- ``dehyphenate`` et ``build_vocabulary`` corrigent le texte LISIBLE extrait des PDF, où des
  césures de fin de ligne ont été saisies comme de vrais tirets (« équipe-ments »).
"""

import re
import unicodedata
from collections.abc import Iterable

# Apostrophes et guillemets typographiques ramenés à leur forme ASCII.
_QUOTES = str.maketrans(
    {
        "’": "'",  # ’ apostrophe typographique
        "‘": "'",  # ‘
        "‛": "'",  # ‛
        "ʼ": "'",  # ʼ lettre apostrophe
        "′": "'",  # ′ prime
        "´": "'",  # ´ accent aigu isolé
        "`": "'",
        "«": '"',  # «
        "»": '"',  # »
        "“": '"',  # “
        "”": '"',  # ”
        "„": '"',  # „
        "‟": '"',  # ‟
        "‹": '"',  # ‹
        "›": '"',  # ›
    }
)
# Tous les tirets : trait d'union, trait d'union insécable, tiret numérique, demi-cadratin,
# cadratin, barre horizontale, signe moins.
_DASHES = re.compile("[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
_WHITESPACE = re.compile(r"\s+")

# Recollage : uniquement entre deux suites de LETTRES (pas de chiffres), pour ne jamais toucher
# aux identifiants exacts comme ERR-402 ou NF C18-510.
_LETTERS = r"[^\W\d_]+"
_HYPHENATED = re.compile(rf"(?<!\w)({_LETTERS})[-\u2010\u2011](?=({_LETTERS})(?!\w))")
_HYPHEN_FREE_WORD = re.compile(_LETTERS)


def normalize_for_match(text: str) -> str:
    """Forme canonique utilisée pour vérifier qu'un extrait cité figure mot pour mot dans sa source.

    C'est la SEULE fonction de comparaison du garde-fou 2 (vérification déterministe des
    citations, ADR 0001) : elle est appliquée aux DEUX côtés, l'extrait cité par le LLM et le
    texte du passage source, puis on teste l'inclusion de l'un dans l'autre. Utiliser une autre
    normalisation d'un seul côté fausserait la vérification.

    Étapes, dans l'ordre :
      1. NFKC (ligatures, espaces et caractères de compatibilité) ;
      2. minuscules ;
      3. apostrophes et guillemets typographiques unifiés (’ → ', « » “ ” → ") ;
      4. suppression des caractères de contrôle et de format, dont le trait d'union
         conditionnel U+00AD et les caractères de mise en page des PDF ;
      5. suppression de TOUS les tirets (-, ‐, ‑, ‒, –, —, ―, −) ;
      6. suppression de TOUS les espaces, insécables compris.

    Les étapes 5 et 6 rendent équivalents « court-circuit », « court circuit » et
    « courtcircuit », ainsi que les césures de fin de ligne, quelle que soit leur forme. Les
    lettres, chiffres et autres signes restent comparés tels quels : un chiffre, une unité ou
    une négation modifiés font toujours échouer la vérification.

    Le résultat n'est pas destiné à l'affichage.
    """
    text = unicodedata.normalize("NFKC", text).lower().translate(_QUOTES)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf"))
    text = _DASHES.sub("", text)
    return _WHITESPACE.sub("", text)


def dehyphenate(text: str, vocabulary: set[str]) -> str:
    """Recolle « mot-suite » en « motsuite » si « motsuite » (minuscules) est dans le vocabulaire.

    Sinon le tiret est conservé (« court-circuit » reste intact si « courtcircuit » n'est pas un
    mot du corpus). Seules les suites de lettres sont concernées : un identifiant contenant des
    chiffres (ERR-402) n'est jamais modifié. La casse d'origine est conservée.
    """

    def join_if_known(match: re.Match[str]) -> str:
        joined = match.group(1) + match.group(2)
        return match.group(1) if joined.lower() in vocabulary else match.group(0)

    return _HYPHENATED.sub(join_if_known, text)


def build_vocabulary(texts: Iterable[str]) -> set[str]:
    """Mots (en minuscules) présents dans le corpus SANS tiret, pour guider ``dehyphenate``.

    Les traits d'union conditionnels (U+00AD) sont retirés avant le découpage : un mot coupé
    par une césure typographique compte comme le mot entier. Un mot n'apparaissant que sous
    une forme à tiret (« court-circuit ») ne produit pas « courtcircuit ».
    """
    vocabulary: set[str] = set()
    for text in texts:
        cleaned = unicodedata.normalize("NFKC", text.replace("\xad", "")).lower()
        vocabulary.update(_HYPHEN_FREE_WORD.findall(cleaned))
    return vocabulary


def quote_matches(quote: str, source: str, min_chars: int = 20) -> bool:
    """Vrai si la citation figure mot pour mot dans la source, une fois les deux normalisées.

    Les deux côtés passent par ``normalize_for_match``. Pourquoi ses règles suffisent à absorber
    les artefacts d'extraction PDF sans accepter une citation fausse :

    - les espaces sont tous supprimés, car l'extraction d'un texte en colonnes coupe les lignes
      et insère des retours à la ligne ou des espaces au milieu d'un mot (« inter médiaire »
      pour « intermédiaire ») ;
    - les tirets et le trait d'union conditionnel U+00AD sont supprimés, car les césures de fin
      de ligne apparaissent tantôt en U+00AD, tantôt en vrai tiret (« inter-médiaire ») ;
    - les caractères de contrôle sont supprimés : la mise en page de l'ED 6109 laisse un
      caractère \x07 devant certains mots et après les puces ;
    - les espaces insécables (U+00A0, U+202F) que la typographie française place devant « ; »,
      « : », « ! » ou « ? » disparaissent avec les autres espaces : « séparation ; » et
      « séparation; » sont équivalents.

    Lettres, chiffres et ponctuation restent comparés : un mot, un chiffre ou une unité
    différents font échouer la vérification.

    Lève ``ValueError`` si la citation normalisée fait moins de ``min_chars`` caractères : une
    citation trop courte se retrouve n'importe où et ne prouve rien.
    """
    needle = normalize_for_match(quote)
    if len(needle) < min_chars:
        raise ValueError(
            f"citation trop courte : {len(needle)} caractères normalisés (minimum {min_chars})"
        )
    return needle in normalize_for_match(source)
