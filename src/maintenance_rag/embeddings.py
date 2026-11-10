"""Calcul des vecteurs d'embedding des passages et des questions via le fournisseur configuré
(dimension imposée par config.yaml et le schéma SQL).

- ``Embedder`` : interface commune (Protocol) ;
- ``OpenAIEmbedder`` : API OpenAI, par lots, avec reprise exponentielle sur erreurs transitoires
  et contrôle de la dimension de chaque vecteur ;
- ``FakeEmbedder`` : vecteurs déterministes dérivés d'un hachage du texte, pour les tests (aucun
  appel réseau).
"""

import hashlib
import math
import struct
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import openai


class Embedder(Protocol):
    """Transforme des textes en vecteurs de dimension ``dimensions``, dans le même ordre."""

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _check_dimensions(vectors: Sequence[Sequence[float]], dimensions: int) -> None:
    for index, vector in enumerate(vectors):
        if len(vector) != dimensions:
            raise ValueError(
                f"vecteur {index} : {len(vector)} dimensions au lieu de {dimensions} "
                "(config.yaml embeddings.dimensions et vector(1536) du schéma SQL)"
            )


# Erreurs pour lesquelles un nouvel essai a un sens (réseau, quota, panne du service). Une
# erreur d'authentification ou de requête invalide échoue tout de suite.
_RETRYABLE: tuple[type[Exception], ...] = (
    openai.APIConnectionError,  # inclut APITimeoutError
    openai.RateLimitError,
    openai.InternalServerError,
)


class OpenAIEmbedder:
    """Embeddings OpenAI : lots de ``batch_size`` textes, ``max_attempts`` essais par lot."""

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        api_key: str,
        batch_size: int = 100,
        max_attempts: int = 3,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if batch_size < 1 or max_attempts < 1:
            raise ValueError("batch_size et max_attempts doivent être positifs")
        self.model = model
        self._dimensions = dimensions
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        # Les reprises sont gérées ici (pas de double reprise par le SDK).
        self._client = client or openai.OpenAI(api_key=api_key, max_retries=0)
        self._sleep = sleep

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.embeddings.create(
                    model=self.model, input=list(batch), dimensions=self._dimensions
                )
                break
            except _RETRYABLE:
                if attempt == self.max_attempts:
                    raise
                self._sleep(2.0 ** (attempt - 1))  # 1 s, 2 s, 4 s…
        data = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in data]
        if len(vectors) != len(batch):
            raise ValueError(f"{len(vectors)} vecteurs reçus pour {len(batch)} textes")
        return vectors

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        _check_dimensions(vectors, self._dimensions)
        return vectors


class FakeEmbedder:
    """Vecteurs déterministes (hachage BLAKE2b du texte), de norme 1 ; aucun appel réseau."""

    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _vector(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self._dimensions:
            digest = hashlib.blake2b(f"{counter}\x00{text}".encode(), digest_size=64).digest()
            values.extend(v / 2**31 - 1.0 for v in struct.unpack("<16I", digest))
            counter += 1
        values = values[: self._dimensions]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = [self._vector(text) for text in texts]
        _check_dimensions(vectors, self._dimensions)
        return vectors
