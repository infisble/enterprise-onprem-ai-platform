"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, 2009).

    RRF(d) = Σ_i  w_i / (k + rank_i(d))        rank starts at 1, k = 60 by default

Why RRF instead of adding raw scores: BM25 scores are unbounded (often 5-40) while cosine
similarities live in [-1, 1]; they are not on comparable scales and their distributions change per
query. RRF uses only *ranks*, needs no score normalisation or training data, and rewards documents
that appear high in BOTH lists. k dampens the influence of the very top ranks (k=60 is the standard
value from the paper; smaller k makes fusion more "winner-takes-all").

Worked example (k=60):
    product A: BM25 rank 1, vector rank 3 -> 1/61 + 1/63 = 0.03226
    product B: BM25 rank 2, not in vector -> 1/62         = 0.01613
    product C: not in BM25, vector rank 1 -> 1/61         = 0.01639
  => A > C > B. A was found by both retrievers, so it wins.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[T, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[T, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], str(x[0])))
