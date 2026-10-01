"""Hybrid product search pipeline.

    query ─► preprocessing + structured filter extraction
          ├─► BM25 (Elasticsearch)  with filters ─┐
          └─► ANN  (Qdrant)         with filters ─┴─► RRF fusion ─► rerank (top 30)
                                                   ─► business rules (live SQL stock, discontinued, lead time)
                                                   ─► ranked products + per-result explanation

Exact product codes (NP-10042, 29904.01, PT-4410-NV) short-circuit to a keyword lookup: a vector
index is the wrong tool for identifiers.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import select

from packages.core.config import get_settings
from packages.core.db import session_scope
from packages.core.models import Product
from packages.core.repository import available_stock, products_by_ids, supplier_codes
from packages.core.security import Permission, Principal
from packages.core.text import expand_query
from packages.observability.tracing import span
from packages.schemas.search import (
    ExtractedQuery,
    ProductHit,
    RankExplanation,
    SearchFilters,
    SearchResponse,
)
from services.llm.embeddings import get_embedder
from services.search import vector
from services.search.documents import embedding_text, product_search_doc
from services.search.filters import extract_query
from services.search.fusion import reciprocal_rank_fusion
from services.search.lexical import get_lexical_index
from services.search.rerank import get_reranker

RELAXABLE = ("category", "material")  # soft constraints we may drop if nothing matches


def _vector_filters_and_search(query_vec: list[float], f: SearchFilters, k: int) -> list[tuple[int, float]]:
    hits = vector.search(get_settings().products_collection, query_vec, vector.product_filter(f), k)
    return [(int(pid), score) for pid, score, _ in hits]


class HybridSearchService:
    def __init__(self) -> None:
        self.settings = get_settings()

    # ------------------------------------------------------------------------------ indexing
    def index_all(self, recreate: bool = True) -> dict[str, int]:
        with session_scope() as s:
            products = s.scalars(select(Product)).all()
            stock = available_stock(s, [p.id for p in products])
            codes = supplier_codes(s)
            docs = [product_search_doc(p, stock[p.id], codes.get(p.supplier_id, "")) for p in products]
        return self.index_docs(docs, recreate=recreate)

    def index_docs(self, docs: list[dict[str, Any]], recreate: bool = False) -> dict[str, int]:
        embedder = get_embedder()
        vector.ensure_collection(
            self.settings.products_collection,
            embedder.dim,
            keyword_fields=["category", "color", "material", "sizes", "supplier", "brand", "status"],
            numeric_fields=["price_eur", "stock_available", "capacity_ml"],
            recreate=recreate,
        )
        vectors = embedder.embed_documents([embedding_text(d) for d in docs])
        vector.upsert(
            self.settings.products_collection,
            [(d["product_id"], v, d) for d, v in zip(docs, vectors, strict=True)],
        )
        n = get_lexical_index().index(docs, recreate=recreate)
        return {"vector": len(docs), "lexical": n}

    def ensure_lexical_loaded(self) -> None:
        """The in-process BM25 fallback lives in memory; rebuild it from SQL on startup."""
        idx = get_lexical_index()
        if idx.backend == "in-process-bm25" and idx.count() == 0:
            with session_scope() as s:
                products = s.scalars(select(Product)).all()
                stock = available_stock(s, [p.id for p in products])
                codes = supplier_codes(s)
                idx.index(
                    [product_search_doc(p, stock[p.id], codes.get(p.supplier_id, "")) for p in products]
                )

    # ------------------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        principal: Principal,
        overrides: SearchFilters | None = None,
        limit: int = 10,
        mode: str = "hybrid",  # hybrid | lexical | vector
    ) -> SearchResponse:
        t0 = time.perf_counter()
        stages: dict[str, dict] = {}
        with span("extract_filters", kind="internal") as a:
            extracted = extract_query(query)
            if overrides:
                merged = extracted.filters.model_dump()
                merged.update({k: v for k, v in overrides.model_dump().items() if v not in (None, "")})
                extracted.filters = SearchFilters(**merged)
            a["filters"] = extracted.filters.active()
        stages["extract"] = {"filters": extracted.filters.active(), "semantic_text": extracted.semantic_text}

        lexical = get_lexical_index()
        if extracted.exact_code:
            with span("exact_code_lookup", kind="retrieval", code=extracted.exact_code):
                ids = lexical.find_code(extracted.exact_code)
            if ids:
                stages["exact_match"] = {"code": extracted.exact_code, "hits": len(ids)}
                hits = self._materialize(
                    ids,
                    {pid: {"bm25_rank": i + 1} for i, pid in enumerate(ids)},
                    {pid: (1.0, ["exact code"]) for pid in ids},
                    principal,
                    extracted,
                )
                return self._response(extracted, hits, len(ids), stages, [], t0)

        relaxed: list[str] = []
        filters = extracted.filters
        while True:
            bm25, vec = self._retrieve(extracted.semantic_text, filters, mode, stages)
            if bm25 or vec or not any(getattr(filters, f) for f in RELAXABLE):
                break
            for f in RELAXABLE:
                if getattr(filters, f):
                    relaxed.append(f"{f}={getattr(filters, f)}")
                    filters = filters.model_copy(update={f: None})
                    break

        with span("rrf_fusion", kind="internal", k=self.settings.rrf_k) as a:
            rankings = [[pid for pid, _ in bm25], [pid for pid, _ in vec]]
            fused = reciprocal_rank_fusion([r for r in rankings if r], k=self.settings.rrf_k)
            a["fused"] = len(fused)
        stages["fusion"] = {"method": f"RRF k={self.settings.rrf_k}", "candidates": len(fused)}

        explain: dict[int, dict[str, Any]] = {}
        for rank, (pid, score) in enumerate(bm25, start=1):
            explain.setdefault(pid, {}).update(bm25_rank=rank, bm25_score=round(score, 3))
        for rank, (pid, score) in enumerate(vec, start=1):
            explain.setdefault(pid, {}).update(vector_rank=rank, vector_score=round(score, 4))
        for pid, score in fused:
            explain.setdefault(pid, {})["rrf_score"] = round(score, 5)

        top = [pid for pid, _ in fused[:30]]
        with span("rerank", kind="rerank", reranker=get_reranker().name, candidates=len(top)):
            t_r = time.perf_counter()
            rerank_scores = self._rerank(extracted.semantic_text, top, explain)
            stages["rerank"] = {
                "model": get_reranker().name,
                "candidates": len(top),
                "ms": round((time.perf_counter() - t_r) * 1000, 2),
            }
        ordered = sorted(top, key=lambda pid: -rerank_scores[pid][0])
        hits = self._materialize(ordered, explain, rerank_scores, principal, extracted)
        return self._response(extracted, hits[:limit], len(fused), stages, relaxed, t0)

    def _retrieve(self, text: str, f: SearchFilters, mode: str, stages: dict) -> tuple[list, list]:
        k = self.settings.search_candidates
        bm25: list[tuple[int, float]] = []
        vec: list[tuple[int, float]] = []
        if mode in ("hybrid", "lexical"):
            with span("bm25", kind="retrieval", backend=get_lexical_index().backend) as a:
                t = time.perf_counter()
                expanded = expand_query(text)
                bm25 = get_lexical_index().search(expanded, f, k)
                a.update(hits=len(bm25), expanded_query=expanded)
            stages["bm25"] = {
                "backend": get_lexical_index().backend,
                "hits": len(bm25),
                "query": expanded,
                "ms": round((time.perf_counter() - t) * 1000, 2),
            }
        if mode in ("hybrid", "vector"):
            with span("vector", kind="retrieval", backend=vector.qdrant_backend()) as a:
                t = time.perf_counter()
                vec = _vector_filters_and_search(get_embedder().embed_query(text), f, k)
                a["hits"] = len(vec)
            stages["vector"] = {
                "backend": vector.qdrant_backend(),
                "embedder": get_embedder().name,
                "hits": len(vec),
                "ms": round((time.perf_counter() - t) * 1000, 2),
            }
        return bm25, vec

    def _rerank(
        self, query: str, ids: list[int], explain: dict[int, dict]
    ) -> dict[int, tuple[float, list[str]]]:
        if not ids:
            return {}
        with session_scope() as s:
            rows = products_by_ids(s, ids)
            cands = []
            for pid in ids:
                p = rows[pid]
                cands.append(
                    {
                        "name_en": p.name_en,
                        "name_sk": p.name_sk,
                        "category": p.category,
                        "subcategory": p.subcategory,
                        "material": p.material,
                        "color": p.color,
                        "tags": p.tags or [],
                        "description_en": p.description_en,
                        "_fused": explain[pid].get("rrf_score", 0.0),
                    }
                )
        scores = get_reranker().score(query, cands)
        return dict(zip(ids, scores, strict=True))

    def _materialize(
        self,
        ids: list[int],
        explain: dict[int, dict],
        scores: dict[int, tuple[float, list[str]]],
        principal: Principal,
        extracted: ExtractedQuery,
    ) -> list[ProductHit]:
        """Business rules against SQL (source of truth), then build response rows."""
        with span("business_rules", kind="sql", candidates=len(ids)), session_scope() as s:
            rows = products_by_ids(s, ids)
            live_stock = available_stock(s, ids)
            codes = supplier_codes(s)
        can_see_stock = principal.can(Permission.STOCK_READ)
        hits: list[ProductHit] = []
        for pid in ids:
            p = rows.get(pid)
            if p is None or p.status != "active":
                continue  # index may be stale; SQL decides
            stock = live_stock.get(pid, 0)
            if extracted.filters.in_stock and stock <= 0:
                continue  # index snapshot said in stock, SQL says sold out
            base, matched = scores.get(pid, (0.0, []))
            adj = (0.05 if stock > 0 else -0.10) + (-0.05 if p.lead_time_days > 21 else 0.0)
            name = p.name_sk if extracted.detected_language in ("sk", "cs") else p.name_en
            hits.append(
                ProductHit(
                    product_id=p.id,
                    sku=p.sku,
                    name=name,
                    name_en=p.name_en,
                    category=p.category,
                    color=p.color,
                    material=p.material,
                    capacity_ml=p.capacity_ml,
                    sizes=p.sizes or [],
                    price_eur=p.price_eur,
                    stock_available=stock if can_see_stock else None,
                    supplier=codes.get(p.supplier_id, ""),
                    brand=p.brand,
                    lead_time_days=p.lead_time_days,
                    tags=p.tags or [],
                    final_score=round(base + adj, 4),
                    explain=RankExplanation(
                        **{
                            k: v for k, v in explain.get(pid, {}).items() if k in RankExplanation.model_fields
                        },
                        rerank_score=base,
                        business_adjustment=adj,
                        matched_attributes=matched,
                    ),
                )
            )
        hits.sort(key=lambda h: -h.final_score)
        if not extracted.filters.category and not extracted.exact_code:
            hits = self._diversify(hits)
        return hits

    @staticmethod
    def _diversify(hits: list[ProductHit], per_group: int = 3) -> list[ProductHit]:
        """Discovery queries ("gift ideas") should not return five colour variants of one item:
        keep at most `per_group` hits per product type in the head of the list, push the rest down."""
        head: list[ProductHit] = []
        tail: list[ProductHit] = []
        seen: dict[str, int] = {}
        for h in hits:
            key = h.name_en.split(",")[0].split()[0] + h.category
            seen[key] = seen.get(key, 0) + 1
            (head if seen[key] <= per_group else tail).append(h)
        return head + tail

    def _response(self, extracted, hits, total, stages, relaxed, t0) -> SearchResponse:
        return SearchResponse(
            query=extracted,
            results=hits,
            total_candidates=total,
            stages=stages,
            relaxed_filters=relaxed,
            lexical_backend=get_lexical_index().backend,
            vector_backend=vector.qdrant_backend(),
            reranker=get_reranker().name,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )


_service: HybridSearchService | None = None


def get_search_service() -> HybridSearchService:
    global _service
    if _service is None:
        _service = HybridSearchService()
    return _service
