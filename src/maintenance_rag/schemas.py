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
