"""Modèles Pydantic du domaine : documents du manifeste, passages (chunks), résultats de recherche,
réponses structurées et citations.
"""

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """Passage indexé : unité de recherche et de citation.

    ``content`` est le texte du document seul (c'est lui que vérifie le garde-fou 2) ;
    ``embedding_text`` le précède de l'en-tête de section pour l'embedding et le reranking.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_slug: str = Field(min_length=1, description="= documents.slug")
    chunk_index: int = Field(ge=0, description="rang dans le document, déterministe")
    section_path: list[str] = Field(description="titres en vigueur, du chapitre au plus fin")
    section_title: str = Field(description="section_path joint par ' › '")
    page_start: int = Field(ge=1, description="page PDF du premier bloc")
    page_end: int = Field(ge=1, description="page PDF du dernier bloc")
    printed_page_start: int | None = Field(description="page imprimée minimale, None si aucune")
    printed_page_end: int | None = Field(description="page imprimée maximale, None si aucune")
    content: str = Field(min_length=1, description="texte du passage, SANS en-tête")
    embedding_text: str = Field(description="en-tête de section + '\\n' + content")
    token_count: int = Field(ge=1, description="tokens tiktoken cl100k_base de embedding_text")
    has_warning: bool = Field(description="le passage contient un « Attention ! »")
    block_types: list[str] = Field(description="types de blocs présents, dans l'ordre")


class Affirmation(BaseModel):
    """Une affirmation de la réponse, adossée à un extrait copié mot pour mot d'un passage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    texte: str = Field(description="affirmation courte, en français")
    extrait_exact: str = Field(
        description="extrait COPIÉ MOT POUR MOT du passage cité (au moins 20 caractères) ; "
        "contient tout chiffre, durée, seuil ou nom présent dans texte"
    )
    passage_id: str = Field(description="identifiant du passage cité : P1, P2…")


class LLMAnswer(BaseModel):
    """Sortie structurée du LLM (JSON schema strict : tous les champs requis, aucun en plus).

    ``info_presente`` : l'information PRÉCISE demandée (valeur, durée, seuil, nom, étape) est
    écrite dans les passages. Si elle est absente, ``reponse`` le dit et ``affirmations`` peut
    donner des informations liées, chacune citée.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    info_presente: bool = Field(
        description="vrai seulement si l'information précise demandée est écrite dans les passages"
    )
    reponse: str = Field(description="réponse courte, en français, pour un technicien")
    affirmations: list[Affirmation] = Field(description="affirmations, chacune citée")


def strict_json_schema(model: type[BaseModel]) -> dict[str, object]:
    """JSON schema du modèle au format « strict » des Structured Outputs : chaque objet liste
    toutes ses propriétés dans ``required`` et interdit les propriétés en plus."""
    schema = model.model_json_schema()

    def tighten(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                tighten(value)
        elif isinstance(node, list):
            for value in node:
                tighten(value)

    tighten(schema)
    return schema
