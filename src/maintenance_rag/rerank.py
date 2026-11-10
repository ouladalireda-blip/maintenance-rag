"""Re-classement des candidats par un modèle cross-encoder multilingue (FlashRank) ; produit le
score utilisé par le garde-fou 1.

- ``Reranker`` : tout objet qui note des passages pour une question (``score``), un score par
  texte, dans l'ordre des textes. ``FlashRankReranker`` (ms-marco-MultiBERT-L-12, local) et
  ``FakeReranker`` (déterministe, pour les tests) en sont deux.
- ``rerank`` : candidats de la recherche → candidats triés par score décroissant
  (``Reranked`` : le ``SearchResult`` intact, son score et son rang de recherche). Égalité de
  score : départage par rang de recherche, puis par chunk_id.
- Texte noté : ``section_title + "\\n" + content`` (contenu seul si le titre est vide), le même
  que ``embedding_text`` du chunking, mesuré sur la fenêtre de 512 tokens. Seule différence :
  le suffixe « (suite) » des morceaux de suite n'est pas stocké en base, donc pas ajouté ici.

Aucun téléchargement : le modèle doit déjà être dans ``rerank.cache_dir`` de config.yaml.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from maintenance_rag.config import CONFIG_PATH, ROOT
from maintenance_rag.retrieval import SearchResult


class Reranker(Protocol):
    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        """Un score par texte (plus haut = plus pertinent), dans l'ordre de ``texts``."""
        ...


@dataclass(frozen=True)
class Reranked:
    """Un candidat re-classé : tous les champs de la recherche sont gardés dans ``result``."""

    result: SearchResult
    score: float
    search_rank: int  # position (dès 1) dans la liste renvoyée par la recherche


def passage_text(result: SearchResult) -> str:
    """Texte transmis au reranker : titre de section + saut de ligne + contenu."""
    return f"{result.section_title}\n{result.content}" if result.section_title else result.content


def rerank(question: str, candidates: Sequence[SearchResult], reranker: Reranker) -> list[Reranked]:
    """Candidats triés par score décroissant ; égalité : rang de recherche, puis chunk_id."""
    if not candidates:
        return []
    scores = reranker.score(question, [passage_text(candidate) for candidate in candidates])
    if len(scores) != len(candidates):
        raise ValueError("le reranker doit renvoyer un score par candidat")
    reranked = [
        Reranked(result=candidate, score=float(score), search_rank=rank)
        for rank, (candidate, score) in enumerate(zip(candidates, scores, strict=True), start=1)
    ]
    return sorted(reranked, key=order_key)


def order_key(reranked: Reranked) -> tuple[float, int, int]:
    """Score décroissant, puis rang de recherche, puis chunk_id. Le rang de recherche étant une
    position, chunk_id n'est qu'une sûreté (listes fusionnées, rangs recopiés)."""
    return (-reranked.score, reranked.search_rank, reranked.result.chunk_id)


class FlashRankReranker:
    """Cross-encoder FlashRank chargé depuis le cache local (jamais téléchargé ici)."""

    def __init__(self, model: str, cache_dir: Path, max_length: int) -> None:
        if not (cache_dir / model).is_dir():
            raise FileNotFoundError(
                f"modèle {model!r} absent de {cache_dir} : aucun téléchargement n'est fait ici"
            )
        from flashrank import Ranker  # type: ignore[import-untyped]  # import tardif (onnxruntime)

        self.model = model
        self.max_length = max_length
        self._ranker: Any = Ranker(
            model_name=model, cache_dir=str(cache_dir), max_length=max_length, log_level="WARNING"
        )

    @classmethod
    def from_config(cls, config_path: Path = CONFIG_PATH) -> "FlashRankReranker":
        """Modèle, cache_dir (relatif à la racine) et max_length de la section ``rerank``."""
        settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))["rerank"]
        cache_dir = Path(settings["cache_dir"])
        return cls(
            model=str(settings["model"]),
            cache_dir=cache_dir if cache_dir.is_absolute() else ROOT / cache_dir,
            max_length=int(settings["max_length"]),
        )

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        from flashrank import RerankRequest

        if not texts:
            return []
        passages = [{"id": index, "text": text} for index, text in enumerate(texts)]
        ranked = self._ranker.rerank(RerankRequest(query=question, passages=passages))
        by_id = {int(item["id"]): float(item["score"]) for item in ranked}
        return [by_id[index] for index in range(len(texts))]


_WORD = re.compile(r"\w+")


class FakeReranker:
    """Reranker déterministe pour les tests : part des mots de la question présents dans le
    texte (sans casse). Garde la trace des appels."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((question, list(texts)))
        words = set(_WORD.findall(question.lower()))
        if not words:
            return [0.0 for _ in texts]
        return [len(words & set(_WORD.findall(text.lower()))) / len(words) for text in texts]
