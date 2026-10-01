from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select

from apps.api.deps import require
from packages.core.db import session_scope
from packages.core.models import ProductUpdateRequest
from packages.core.repository import available_stock, price_out, product_by_sku, product_out, stock_out
from packages.core.security import Permission, Principal
from packages.observability.tracing import start_trace
from packages.schemas.search import SearchFilters, SearchResponse
from services.agent.service import persist_trace
from services.search.hybrid import get_search_service
from services.search.lexical import get_lexical_index

router = APIRouter(prefix="/api", tags=["catalog"])


@router.get("/search/products", response_model=SearchResponse)
async def search_products(
    q: str = Query(min_length=1, max_length=300),
    mode: str = Query("hybrid", pattern="^(hybrid|lexical|vector)$"),
    limit: int = Query(12, ge=1, le=50),
    price_min: float | None = None,
    price_max: float | None = None,
    category: str | None = None,
    color: str | None = None,
    material: str | None = None,
    in_stock: bool | None = None,
    supplier: str | None = None,
    brand: str | None = None,
    size: str | None = None,
    language: str | None = None,
    principal: Principal = Depends(require(Permission.PRODUCTS_SEARCH)),
) -> SearchResponse:
    overrides = SearchFilters(
        price_min=price_min,
        price_max=price_max,
        category=category,
        color=color,
        material=material,
        in_stock=in_stock,
        supplier=supplier,
        brand=brand,
        size=size,
        language=language,
    )

    def run() -> SearchResponse:
        with start_trace(
            username=principal.username, role=principal.role.value, kind="search", query=q
        ) as trace:
            trace.route = f"search:{mode}"
            res = get_search_service().search(q, principal, overrides, limit=limit, mode=mode)
            persist_trace(trace, kind="search")
            return res

    return await run_in_threadpool(run)


@router.get("/search/facets")
def facets(principal: Principal = Depends(require(Permission.PRODUCTS_SEARCH))) -> dict:
    return get_lexical_index().facets(SearchFilters())


@router.get("/products/{sku}")
def get_product(sku: str, principal: Principal = Depends(require(Permission.PRODUCTS_READ))) -> dict:
    with session_scope() as s:
        p = product_by_sku(s, sku)
        if p is None:
            raise HTTPException(404, "Product not found")
        can_stock = principal.can(Permission.STOCK_READ)
        out = product_out(p, available_stock(s, [p.id])[p.id] if can_stock else None).model_dump()
        out["stock"] = stock_out(s, p).model_dump() if can_stock else None
        out["pricing"] = price_out(s, p).model_dump() if principal.can(Permission.PRICING_READ) else None
        out["update_requests"] = [
            {
                "id": r.id,
                "field": r.field,
                "status": r.status,
                "requested_by": r.requested_by,
                "requested_at": r.requested_at.isoformat(),
            }
            for r in s.scalars(
                select(ProductUpdateRequest)
                .where(ProductUpdateRequest.product_id == p.id)
                .order_by(ProductUpdateRequest.id.desc())
                .limit(10)
            )
        ]
        return out
