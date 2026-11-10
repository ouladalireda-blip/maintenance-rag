"""Extraction et normalisation des références exactes (codes d'erreur, références de pièces, ex.
ERR-402, FILTRE-XJ402) dans les textes et les questions.

Une référence est réduite à une forme CANONIQUE (``canonical``) : majuscules, sans espaces ni
points, tirets et barres obliques conservés. La même fonction s'applique aux passages à
l'indexation et aux questions à la recherche : « R. 4411-6 », « R.4411-6 » et
« article R 4411-6 » donnent tous « R4411-6 » et se retrouvent par égalité stricte (colonne
``chunks.identifiers`` et classement ``identifier_match`` de hybrid_search).

Familles reconnues (une expression régulière par famille) : normes NF / EN / ISO / IEC,
articles de norme, articles du Code du travail, règlements et directives européens,
brochures INRS, procédés d'isolement R1 à R4 (seulement en contexte).
"""

import re
import unicodedata

_SPACE = r"[    ]"  # espace, insécable, fine insécable, fine

# Norme : un ou plusieurs préfixes d'organisme (NF, EN, ISO, IEC/CEI) suivis d'un numéro,
# alphanumérique à la française (« C18-510 », « C15-100 ») ou numérique avec partie
# éventuelle (« 14118 », « 60204-1 », « 12266-1 »). EN doit être en majuscules : « en 2015 »
# est du français. Ex. NF C18-510, NF EN ISO 14118, EN 60204-1.
_STANDARD_BODY = r"(?:(?i:NF|ISO|IEC|CEI)|EN)"
_STANDARD = (
    rf"(?<![\w-]){_STANDARD_BODY}(?:{_SPACE}+{_STANDARD_BODY})*{_SPACE}+"
    r"(?:[A-Z]\s?\d{2}-\d{3}|\d{3,5}(?:-\d{1,2})?)(?![\w-])"
)
STANDARD = re.compile(_STANDARD)

# Article de norme, dans les deux ordres usuels :
#   « article 536.2 de la norme NF C15-100 » ; « la norme NF C15-100, article 536.2 ».
_ARTICLE_NUMBER = r"\d+(?:\.\d+)+"
STANDARD_ARTICLE = re.compile(
    rf"(?i:article){_SPACE}+(?P<article>{_ARTICLE_NUMBER}){_SPACE}+(?i:de{_SPACE}+la{_SPACE}+norme)"
    rf"{_SPACE}+(?P<standard>{_STANDARD})"
    rf"|(?i:norme){_SPACE}+(?P<standard2>{_STANDARD}){_SPACE}*,?{_SPACE}*(?i:article){_SPACE}+"
    rf"(?P<article2>{_ARTICLE_NUMBER})"
)

# Article du Code du travail : partie réglementaire (R), législative (L) ou décret (D), point
# facultatif, numéro à 4 chiffres et tiret. Ex. R. 4411-6, R.4411-6, R 4544-11, L. 4121-1.
LABOUR_CODE = re.compile(rf"(?<![\w.])[RLD]\.?{_SPACE}?\d{{4}}-\d{{1,3}}(?:-\d{{1,3}})?(?![\w-])")

# Directive européenne : année / numéro / CE, CEE, UE ou EU. Ex. 2006/42/CE.
EU_DIRECTIVE = re.compile(r"(?<![\w/])(?:19|20)\d{2}/\d{1,4}/(?:CEE|CE|UE|EU)(?![\w/])")
# Règlement européen : numéro de 3 ou 4 chiffres / année. Ex. 1272/2008. Exiger 3 chiffres au
# moins écarte les dates (« 10/11/2026 » ne contient que « 11/2026 »).
EU_REGULATION = re.compile(r"(?<![\w/])\d{3,4}/(?:19|20)\d{2}(?![\w/])")
# Règlements européens désignés par leur sigle. Ex. règlement « CLP ».
EU_REGULATION_ACRONYM = re.compile(r"(?<![\w])(?:CLP|REACH)(?![\w])")

# Brochure INRS : « ED » suivi de 3 ou 4 chiffres. Ex. ED 6109, ED 6129, ED 941.
INRS_BROCHURE = re.compile(rf"(?<![\w]){'ED'}{_SPACE}?\d{{3,4}}(?![\w-])")

# Procédés d'isolement R1 à R4 : trop ambigus seuls (« R1 » peut être un repère de schéma) ;
# retenus seulement après « procédé » ou « isolement », dans la même proposition (avant le
# premier « . », « ; » ou retour à la ligne, 60 caractères au plus), éventuellement entre
# parenthèses ou guillemets. Ex. « procédé d'isolement renforcé R1 », « (R2) ».
ISOLATION_CONTEXT = re.compile(r"(?i:proc[ée]d[ée]s?|isolements?)")
ISOLATION_PROCESS = re.compile(r"(?<![\w.])R[1-4](?![\w.-])")
_CONTEXT_WINDOW = 60
_CONTEXT_STOP = re.compile(r"[.;\n]")
_LEADING_ARTICLE = re.compile(r"^ARTICLES?(?=[\s   ]|[RLD])")


def canonical(raw: str) -> str:
    """Forme canonique d'une référence : majuscules, sans espaces ni points, tiret conservé.

    Le mot « article » de tête est retiré : « article R 4411-6 » → « R4411-6 ».
    Ex. « NF C18-510 » → « NFC18-510 » ; « R. 4411-6 » → « R4411-6 ».
    """
    text = unicodedata.normalize("NFKC", raw).upper().strip()
    text = _LEADING_ARTICLE.sub("", text)
    return re.sub(r"[\s.]", "", text)


def _isolation_processes(text: str) -> list[str]:
    found: list[str] = []
    for context in ISOLATION_CONTEXT.finditer(text):
        window = text[context.end() : context.end() + _CONTEXT_WINDOW]
        stop = _CONTEXT_STOP.search(window)
        found.extend(ISOLATION_PROCESS.findall(window[: stop.start()] if stop else window))
    return found


def extract_identifiers(text: str) -> list[str]:
    """Références exactes du texte, sous forme canonique, sans doublon, triées."""
    text = unicodedata.normalize("NFKC", text)
    found: set[str] = set()
    for match in STANDARD_ARTICLE.finditer(text):
        standard = canonical(match.group("standard") or match.group("standard2"))
        article = match.group("article") or match.group("article2")
        found.add(f"{standard}:{article}")  # le numéro d'article garde ses points
    for pattern in (
        STANDARD,
        LABOUR_CODE,
        EU_DIRECTIVE,
        EU_REGULATION,
        EU_REGULATION_ACRONYM,
        INRS_BROCHURE,
    ):
        found.update(canonical(match.group(0)) for match in pattern.finditer(text))
    found.update(_isolation_processes(text))
    return sorted(found)
