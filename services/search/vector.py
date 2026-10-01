"""Qdrant vector store for product and document embeddings (ADR-001).

Qdrant is used for what it does well: ANN search with *payload filtering applied inside the index*
(price, colour, stock ... for products; `access_roles` for document chunks). The same client code
works against the Qdrant server (Docker / on-prem) and qdrant-client's embedded local mode.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from packages.core.config import get_settings
from packages.schemas.search import SearchFilters
from services.search.lexical import material_values

_client: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        s = get_settings()
        if s.qdrant_url:
            _client = QdrantClient(url=s.qdrant_url, timeout=30)
        elif s.qdrant_local_path == ":memory:":
            _client = QdrantClient(location=":memory:")
        else:
            _client = QdrantClient(path=s.qdrant_local_path)
    return _client


def reset_qdrant() -> None:
    global _client
    if _client is not None:
        with contextlib.suppress(Exception):
            _client.close()
    _client = None


def qdrant_backend() -> str:
    s = get_settings()
    return "qdrant-server" if s.qdrant_url else "qdrant-embedded"


def ensure_collection(
    name: str, dim: int, keyword_fields: list[str], numeric_fields: list[str], recreate: bool = False
) -> None:
    client = get_qdrant()
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(name, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))
        if get_settings().qdrant_url:  # payload indexes are a server feature (no-ops locally)
            for f in keyword_fields:
                client.create_payload_index(name, f, qm.PayloadSchemaType.KEYWORD)
            for f in numeric_fields:
                client.create_payload_index(name, f, qm.PayloadSchemaType.FLOAT)


def upsert(
    name: str, points: Sequence[tuple[int | str, list[float], dict[str, Any]]], batch: int = 256
) -> None:
    client = get_qdrant()
    for i in range(0, len(points), batch):
        client.upsert(
            name,
            points=[
                qm.PointStruct(id=pid, vector=vec, payload=payload)
                for pid, vec, payload in points[i : i + batch]
            ],
        )


def chunk_point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def product_filter(f: SearchFilters) -> qm.Filter:
    must: list[qm.Condition] = [qm.FieldCondition(key="status", match=qm.MatchValue(value="active"))]
    if f.price_min is not None or f.price_max is not None:
        must.append(qm.FieldCondition(key="price_eur", range=qm.Range(gte=f.price_min, lte=f.price_max)))
    if f.capacity_min_ml is not None or f.capacity_max_ml is not None:
        must.append(
            qm.FieldCondition(key="capacity_ml", range=qm.Range(gte=f.capacity_min_ml, lte=f.capacity_max_ml))
        )
    for key, value in (
        ("category", f.category),
        ("color", f.color),
        ("sizes", f.size),
        ("supplier", f.supplier),
        ("brand", f.brand.lower() if f.brand else None),
    ):
        if value:
            must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
    if mats := material_values(f.material):
        must.append(qm.FieldCondition(key="material", match=qm.MatchAny(any=mats)))
    if f.in_stock:
        must.append(qm.FieldCondition(key="stock_available", range=qm.Range(gt=0)))
    return qm.Filter(must=must)


def search(name: str, vector: list[float], flt: qm.Filter | None, k: int) -> list[tuple[Any, float, dict]]:
    res = get_qdrant().query_points(name, query=vector, query_filter=flt, limit=k, with_payload=True)
    return [(p.id, float(p.score), p.payload or {}) for p in res.points]


def count(name: str) -> int:
    client = get_qdrant()
    if not client.collection_exists(name):
        return 0
    return int(client.count(name, exact=True).count)
