"""Génération de la réponse structurée (réponse + affirmations citées) à partir des passages
retenus, via le client LLM.

- ``SYSTEM_PROMPT`` / ``PROMPT_VERSION`` : prompt système versionné dans le code. Toute
  modification du texte impose de changer ``PROMPT_VERSION`` (enregistrée avec chaque résultat).
- ``label_passages`` : les passages reçoivent un id court P1…Pn, dans l'ordre de la recherche.
- ``build_user_message`` : question + passages (id, titre de section, page imprimée, texte).
- ``generate`` : appel du LLM avec le JSON schema strict de ``LLMAnswer``, puis validation
  Pydantic. Une sortie incomplète ou invalide ne lève pas d'exception : ``output`` vaut ``None``
  et ``error`` dit pourquoi (le garde-fou 2 en fera un refus). Les erreurs réseau remontent.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from maintenance_rag.llm import LLM, LLMReply
from maintenance_rag.retrieval import SearchResult
from maintenance_rag.schemas import LLMAnswer, strict_json_schema

PROMPT_VERSION = "generation-v1-2026-11-11"

SYSTEM_PROMPT = """\
Tu es l'assistant d'un technicien de maintenance. Tu réponds à sa question à partir d'extraits \
d'un guide de prévention (les « passages »), fournis avec la question.

Règles, toutes obligatoires :
1. Réponds UNIQUEMENT à partir des passages fournis. N'utilise jamais tes connaissances \
générales, même si tu crois connaître la réponse.
2. Chaque affirmation a un extrait_exact COPIÉ MOT POUR MOT d'un seul passage (au moins \
20 caractères ; sans reformuler, sans corriger, sans couper par « … ») et le passage_id de ce \
passage (P1, P2…).
3. Tout chiffre, durée, seuil, référence ou nom présent dans le texte d'une affirmation doit \
figurer dans son extrait_exact.
4. info_presente = true seulement si l'information PRÉCISE demandée (valeur, durée, seuil, nom, \
étape, référence) est écrite explicitement dans les passages. Sinon, info_presente = false : ne \
déduis pas, n'extrapole pas, ne complète pas. La question peut citer une référence (article, \
norme, numéro) : si cette référence n'est pas écrite dans les passages, info_presente = false.
5. Si info_presente = false, reponse dit que les passages fournis ne contiennent pas cette \
information ; les affirmations peuvent donner des informations liées, chacune citée.
6. reponse est courte, en français, pour un technicien. Elle ne contient aucun chiffre, durée, \
seuil ou nom qui ne figure pas dans une affirmation.
"""

SCHEMA_NAME = "reponse_maintenance"
SCHEMA = strict_json_schema(LLMAnswer)


@dataclass(frozen=True)
class Generation:
    output: LLMAnswer | None  # None : sortie incomplète ou invalide
    error: str | None
    reply: LLMReply
    passages: dict[str, SearchResult]  # id court → passage fourni
    prompt_version: str = PROMPT_VERSION


def label_passages(passages: Sequence[SearchResult]) -> dict[str, SearchResult]:
    """Ids courts P1…Pn, dans l'ordre reçu."""
    return {f"P{index}": passage for index, passage in enumerate(passages, start=1)}


def printed_pages(passage: SearchResult) -> str:
    start, end = passage.printed_page_start, passage.printed_page_end
    if start is None or end is None:
        return "page non numérotée"
    return f"page {start}" if start == end else f"pages {start}-{end}"


def build_user_message(question: str, labelled: dict[str, SearchResult]) -> str:
    blocks = [
        f"[{label}] Section : {passage.section_title or '(sans titre)'} — "
        f"{printed_pages(passage)}\n{passage.content}"
        for label, passage in labelled.items()
    ]
    return f"Question : {question}\n\nPassages :\n\n" + "\n\n".join(blocks)


def parse(reply: LLMReply) -> tuple[LLMAnswer | None, str | None]:
    """Sortie validée, ou ``None`` et la raison (sortie incomplète, JSON invalide)."""
    if reply.status != "completed":
        return None, f"sortie incomplète (statut {reply.status})"
    try:
        return LLMAnswer.model_validate_json(reply.text), None
    except ValidationError as error:
        return None, f"sortie invalide ({error.error_count()} erreur(s) de schéma)"


def generate(question: str, passages: Sequence[SearchResult], llm: LLM) -> Generation:
    labelled = label_passages(passages)
    reply = llm.complete(SYSTEM_PROMPT, build_user_message(question, labelled), SCHEMA, SCHEMA_NAME)
    output, error = parse(reply)
    return Generation(output=output, error=error, reply=reply, passages=labelled)
