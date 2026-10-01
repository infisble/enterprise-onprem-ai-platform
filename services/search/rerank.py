"""Reranker abstraction.

First-stage retrieval (BM25 + ANN + RRF) is optimised for recall; the reranker re-orders the top ~30
candidates for precision by looking at query and candidate *together*.

* `FeatureReranker` (default): deterministic, explainable, ~0.1 ms/candidate. Scores attribute
  coverage (query terms/concepts present in the product) blended with the fused retrieval score.
* `CrossEncoderReranker`: local cross-encoder, e.g. BAAI/bge-reranker-v2-m3 (multilingual) on GPU/CPU.
* `HttpReranker`: Hugging Face TEI `/rerank` endpoint on the GPU server.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import httpx

from packages.core.config import get_settings
from packages.core.text import CONCEPT_LOOKUP, stem, tokenize


class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, candidates: list[dict[str, Any]]) -> list[tuple[float, list[str]]]:
        """Return (score in 0..1, matched attribute labels) per candidate, same order."""


def _candidate_text(c: dict[str, Any]) -> str:
    return " ".join(
        str(x)
        for x in (
            c.get("name_en"),
            c.get("name_sk"),
            c.get("category"),
            c.get("subcategory"),
            c.get("material"),
            c.get("color"),
            " ".join(c.get("tags", [])),
            c.get("description_en", ""),
        )
        if x
    )


class FeatureReranker(Reranker):
    name = "feature-reranker-v1"

    def score(self, query: str, candidates: list[dict[str, Any]]) -> list[tuple[float, list[str]]]:
        q_tokens = tokenize(query)
        q_terms = {stem(t) for t in q_tokens}
        q_concepts = {CONCEPT_LOOKUP[t] for t in q_tokens if t in CONCEPT_LOOKUP}
        max_fused = max((c.get("_fused", 0.0) for c in candidates), default=1.0) or 1.0
        out = []
        for c in candidates:
            c_tokens = tokenize(_candidate_text(c))
            c_terms = {stem(t) for t in c_tokens}
            c_concepts = {CONCEPT_LOOKUP[t] for t in c_tokens if t in CONCEPT_LOOKUP}
            matched_terms = sorted(q_terms & c_terms)
            matched_concepts = sorted(q_concepts & c_concepts)
            term_cov = len(matched_terms) / len(q_terms) if q_terms else 0.0
            concept_cov = len(matched_concepts) / len(q_concepts) if q_concepts else term_cov
            fused = c.get("_fused", 0.0) / max_fused
            s = 0.35 * term_cov + 0.40 * concept_cov + 0.25 * fused
            out.append((round(s, 4), matched_terms[:6] + [f"concept:{x}" for x in matched_concepts]))
        return out


class CrossEncoderReranker(Reranker):  # pragma: no cover - optional heavy dependency
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model
        self._model = CrossEncoder(model)

    def score(self, query: str, candidates: list[dict[str, Any]]) -> list[tuple[float, list[str]]]:
        import math

        raw = self._model.predict([(query, _candidate_text(c)) for c in candidates])
        return [(1 / (1 + math.exp(-float(s))), []) for s in raw]


class HttpReranker(Reranker):  # pragma: no cover - requires TEI server
    def __init__(self, url: str, model: str) -> None:
        self.name = f"tei:{model}"
        self._url = url
        self._client = httpx.Client(timeout=30)

    def score(self, query: str, candidates: list[dict[str, Any]]) -> list[tuple[float, list[str]]]:
        r = self._client.post(
            self._url, json={"query": query, "texts": [_candidate_text(c) for c in candidates]}
        )
        r.raise_for_status()
        scores = {item["index"]: float(item["score"]) for item in r.json()}
        return [(scores.get(i, 0.0), []) for i in range(len(candidates))]


@lru_cache
def get_reranker() -> Reranker:
    s = get_settings()
    if s.reranker_provider == "cross_encoder":
        return CrossEncoderReranker(s.reranker_model)
    if s.reranker_provider == "http":
        return HttpReranker(s.reranker_url, s.reranker_model)
    return FeatureReranker()
