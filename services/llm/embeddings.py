"""Embedding providers.

* `hash` (default, offline): signed feature hashing of stemmed tokens, character trigrams and
  domain *concepts* (eco, event, tech, ...) from `packages.core.text.CONCEPTS`. Deterministic, no
  model download, good enough to demonstrate semantic-ish retrieval and the Qdrant integration.
* `openai_compatible`: any `/v1/embeddings` server - Hugging Face TEI or vLLM serving
  `intfloat/multilingual-e5-*` / `BAAI/bge-m3` on the GPU box, or OpenAI.
* `fastembed`: local ONNX multilingual model on CPU (optional dependency).

Production recommendation for SK/CS content: `BAAI/bge-m3` or `multilingual-e5-large` (see ADR-001).
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx
import numpy as np

from packages.core.config import get_settings
from packages.core.text import (
    COLOR_LOOKUP,
    CONCEPT_LOOKUP,
    MATERIAL_SYNONYMS,
    product_type,
    stem,
    tokenize,
)


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashingEmbedder(Embedder):
    name = "hash-concept-v1"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _add(self, vec: np.ndarray, feature: str, weight: float) -> None:
        h = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        vec[idx] += sign * weight

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        for tok in tokens:
            s = stem(tok)
            self._add(vec, f"w:{s}", 1.0)
            if tok in CONCEPT_LOOKUP:
                self._add(vec, f"c:{CONCEPT_LOOKUP[tok]}", 1.6)
            if s in CONCEPT_LOOKUP:
                self._add(vec, f"c:{CONCEPT_LOOKUP[s]}", 0.8)
            if tok in COLOR_LOOKUP:
                self._add(vec, f"color:{COLOR_LOOKUP[tok]}", 1.2)
            if tok in MATERIAL_SYNONYMS:
                self._add(vec, f"mat:{MATERIAL_SYNONYMS[tok]}", 1.2)
            if ptype := product_type(tok):
                cat, sub = ptype
                self._add(vec, f"type:{sub}", 1.5)
                self._add(vec, f"cat:{cat}", 0.8)
            padded = f"#{s}#"
            for i in range(len(padded) - 2):
                self._add(vec, f"g:{padded[i : i + 3]}", 0.25)
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class OpenAICompatibleEmbedder(Embedder):
    def __init__(self, base_url: str, model: str, api_key: str, dim: int) -> None:
        self.name = model
        self.dim = dim
        self._url = base_url.rstrip("/") + "/embeddings"
        self._client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {api_key}"})
        self._e5 = "e5" in model.lower()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            r = self._client.post(self._url, json={"model": self.name, "input": texts[i : i + 64]})
            r.raise_for_status()
            out.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([f"passage: {t}" if self._e5 else t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"query: {text}" if self._e5 else text])[0]


class FastEmbedEmbedder(Embedder):  # pragma: no cover - optional dependency
    def __init__(self, model: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        self.name = model
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_provider == "openai_compatible":
        return OpenAICompatibleEmbedder(
            s.embedding_base_url, s.embedding_model, s.embedding_api_key, s.embedding_dim
        )
    if s.embedding_provider == "fastembed":
        return FastEmbedEmbedder(s.embedding_model)
    return HashingEmbedder(s.embedding_dim)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
