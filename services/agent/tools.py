"""Typed business tools exposed to the agent (the ONLY way the agent touches data).

Note what is *not* here: no `execute_sql`, no `export_customers`, no `send_email`, no direct product
update. The absence of a capability is the strongest defence against prompt injection.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select

from packages.core.db import session_scope
from packages.core.models import Product, Supplier
from packages.core.repository import (
    available_stock,
    price_out,
    product_by_sku,
    product_out,
    stock_out,
)
from packages.core.security import Permission, Principal
from packages.schemas.search import SearchFilters
from services.agent import approvals
from services.agent.analytics_sql import run_analytics_query
from services.agent.registry import ToolError, ToolRegistry, ToolSpec
from services.agent.validation import validate_description
from services.llm.factory import get_llm
from services.llm.prompts import description_messages, translation_messages
from services.rag.service import get_rag_service
from services.search.hybrid import get_search_service

SKU = Field(
    min_length=3,
    max_length=32,
    pattern=r"^[A-Za-z0-9.\-]+$",
    description="NovaPromo SKU (NP-10042) or supplier code",
)
Lang = Literal["sk", "cs", "en"]


# ---------------------------------------------------------------------------------- schemas
class SearchProductsIn(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    price_max: float | None = Field(default=None, gt=0)
    color: str | None = None
    in_stock: bool | None = None
    limit: int = Field(default=8, ge=1, le=25)


class SkuIn(BaseModel):
    sku: str = SKU


class PriceIn(BaseModel):
    sku: str = SKU
    quantity: int = Field(default=1, ge=1, le=1_000_000)


class DocsIn(BaseModel):
    query: str = Field(min_length=3, max_length=500)


class SupplierIn(BaseModel):
    supplier: str | None = Field(
        default=None, max_length=64, description="supplier code (EGT/PTX/ECO) or name"
    )
    sku: str | None = Field(default=None, max_length=32)


class GenerateIn(BaseModel):
    sku: str = SKU
    language: Lang = "sk"
    guidelines: list[str] = Field(default_factory=list, max_length=20)


class TranslateIn(BaseModel):
    sku: str = SKU
    source: Lang = "sk"
    target: Lang = "cs"
    text: str | None = Field(default=None, max_length=4000)


class ValidateIn(BaseModel):
    sku: str = SKU
    language: Lang = "sk"
    text: str = Field(min_length=1, max_length=4000)


class PrepareUpdateIn(BaseModel):
    sku: str = SKU
    field: Literal["description_sk", "description_cs", "description_en", "name_sk", "name_cs"]
    new_value: str = Field(min_length=10, max_length=4000)
    rationale: str = Field(default="", max_length=500)
    validation: dict[str, Any] = Field(default_factory=dict)


class RequestIdIn(BaseModel):
    request_id: int = Field(ge=1)
    note: str = Field(default="", max_length=500)


class AnalyticsIn(BaseModel):
    sql: str = Field(min_length=6, max_length=4000)


# ---------------------------------------------------------------------------------- impls
def _product_or_error(session, sku: str) -> Product:
    p = product_by_sku(session, sku)
    if p is None:
        raise ToolError(f"Product {sku} not found")
    return p


def search_products(a: SearchProductsIn, principal: Principal) -> dict[str, Any]:
    overrides = SearchFilters(price_max=a.price_max, color=a.color, in_stock=a.in_stock)
    res = get_search_service().search(a.query, principal, overrides=overrides, limit=a.limit)
    return res.model_dump()


def get_product(a: SkuIn, principal: Principal) -> dict[str, Any]:
    with session_scope() as s:
        p = _product_or_error(s, a.sku)
        stock = available_stock(s, [p.id])[p.id] if principal.can(Permission.STOCK_READ) else None
        return product_out(p, stock).model_dump()


def get_product_stock(a: SkuIn, _principal: Principal) -> dict[str, Any]:
    with session_scope() as s:
        return stock_out(s, _product_or_error(s, a.sku)).model_dump()


def get_product_price(a: PriceIn, _principal: Principal) -> dict[str, Any]:
    with session_scope() as s:
        price = price_out(s, _product_or_error(s, a.sku))
    unit = price.list_price_eur
    for tier in price.tiers:
        if a.quantity >= tier["min_qty"]:
            unit = tier["unit_price_eur"]
    return {
        **price.model_dump(),
        "quantity": a.quantity,
        "unit_price_eur": unit,
        "total_eur": round(unit * a.quantity, 2),
        "vat_note": "prices exclude VAT",
    }


def search_company_documents(a: DocsIn, principal: Principal) -> dict[str, Any]:
    rag = get_rag_service()
    chunks, filtered = rag.retrieve(a.query, principal)
    return {"chunks": [c.__dict__ for c in chunks], "filtered_by_rbac": filtered}


def get_supplier_info(a: SupplierIn, _principal: Principal) -> dict[str, Any]:
    with session_scope() as s:
        sup = None
        if a.sku:
            p = _product_or_error(s, a.sku)
            sup = s.get(Supplier, p.supplier_id)
        elif a.supplier:
            key = a.supplier.strip().lower()
            for candidate in s.scalars(select(Supplier)):
                if key in (candidate.code.lower(), candidate.name.lower()) or key in candidate.name.lower():
                    sup = candidate
                    break
        if sup is None:
            raise ToolError("Supplier not found")
        return {
            "code": sup.code,
            "name": sup.name,
            "country": sup.country,
            "feed_format": sup.feed_format,
            "default_lead_time_days": sup.default_lead_time_days,
            "urgent_delivery_supported": sup.urgent_delivery_supported,
            "contact_email": sup.contact_email,
            "notes": sup.notes,
        }


def _product_facts(p: Product) -> dict[str, Any]:
    return {
        "sku": p.sku,
        "name_sk": p.name_sk,
        "name_cs": p.name_cs,
        "name_en": p.name_en,
        "category": p.category,
        "subcategory": p.subcategory,
        "material": p.material,
        "color": p.color,
        "capacity_ml": p.capacity_ml,
        "sizes": p.sizes or [],
        "tags": p.tags or [],
        "description_en": p.description_en,
    }


def generate_product_description(a: GenerateIn, _principal: Principal) -> dict[str, Any]:
    with session_scope() as s:
        facts = _product_facts(_product_or_error(s, a.sku))
    guidelines = a.guidelines or [
        "350-600 characters",
        "no exclamation marks or superlatives",
        "mention material, capacity/size and colour",
        "end with branding options",
    ]
    llm = get_llm()
    result = llm.complete(
        description_messages(facts, guidelines, a.language),
        task="product_description",
        payload={"product": facts, "language": a.language, "guidelines": guidelines},
    )
    return {
        "sku": facts["sku"],
        "language": a.language,
        "text": result.text.strip(),
        "model": result.model,
        "provider": result.provider,
        "tokens": result.output_tokens,
    }


def translate_product_description(a: TranslateIn, _principal: Principal) -> dict[str, Any]:
    with session_scope() as s:
        p = _product_or_error(s, a.sku)
        source_text = a.text or getattr(p, f"description_{a.source}")
    if not source_text:
        raise ToolError(f"{a.sku} has no {a.source} description to translate")
    result = get_llm().complete(
        translation_messages(source_text, a.source, a.target),
        task="translate",
        payload={"text": source_text, "source": a.source, "target": a.target},
    )
    return {"sku": a.sku, "source": a.source, "target": a.target, "text": result.text.strip()}


def validate_product_data(a: ValidateIn, _principal: Principal) -> dict[str, Any]:
    with session_scope() as s:
        facts = _product_facts(_product_or_error(s, a.sku))
    return validate_description(a.text, a.language, facts)


def prepare_product_update(a: PrepareUpdateIn, principal: Principal) -> dict[str, Any]:
    try:
        req = approvals.create_request(
            principal, a.sku.upper(), a.field, a.new_value, a.rationale, a.validation
        )
    except approvals.ApprovalError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "request_id": req.id,
        "status": req.status,
        "sku": req.sku,
        "field": req.field,
        "payload_hash": req.payload_hash,
        "base_version": req.base_version,
        "requested_by": req.requested_by,
    }


def approve_product_update(a: RequestIdIn, principal: Principal) -> dict[str, Any]:
    try:
        req = approvals.approve(a.request_id, principal, a.note)
    except approvals.ApprovalError as exc:
        raise ToolError(str(exc)) from exc
    return {"request_id": req.id, "status": req.status, "approved_by": req.decided_by}


def publish_product_update(a: RequestIdIn, principal: Principal) -> dict[str, Any]:
    try:
        req = approvals.publish(a.request_id, principal)
    except approvals.ApprovalError as exc:
        raise ToolError(str(exc)) from exc
    return {"request_id": req.id, "status": req.status}


def analytics_query(a: AnalyticsIn, principal: Principal) -> dict[str, Any]:
    result = run_analytics_query(a.sql, principal)
    if result["outcome"] != "success":
        raise ToolError(f"Query rejected: {result['error']}")
    return result


def build_registry() -> ToolRegistry:
    r = ToolRegistry()
    P = Permission
    for spec in [
        ToolSpec(
            "search_products",
            "Hybrid product search (BM25 + vectors + filters). Use for finding products.",
            SearchProductsIn,
            P.PRODUCTS_SEARCH,
            "read",
            search_products,
        ),
        ToolSpec(
            "get_product", "Get one product's details by SKU.", SkuIn, P.PRODUCTS_READ, "read", get_product
        ),
        ToolSpec(
            "get_product_stock",
            "Current available stock per warehouse for a SKU (from SQL).",
            SkuIn,
            P.STOCK_READ,
            "read",
            get_product_stock,
        ),
        ToolSpec(
            "get_product_price",
            "List price and volume price tiers for a SKU and quantity.",
            PriceIn,
            P.PRICING_READ,
            "read",
            get_product_price,
        ),
        ToolSpec(
            "search_company_documents",
            "Search internal documents (policies, guidelines, terms).",
            DocsIn,
            P.DOCS_READ,
            "read",
            search_company_documents,
        ),
        ToolSpec(
            "get_supplier_info",
            "Supplier master data: lead time, express delivery, contacts.",
            SupplierIn,
            P.SUPPLIERS_READ,
            "read",
            get_supplier_info,
        ),
        ToolSpec(
            "generate_product_description",
            "Draft a product description (does not save anything).",
            GenerateIn,
            P.CONTENT_GENERATE,
            "read",
            generate_product_description,
        ),
        ToolSpec(
            "translate_product_description",
            "Translate a product description (does not save).",
            TranslateIn,
            P.CONTENT_GENERATE,
            "read",
            translate_product_description,
        ),
        ToolSpec(
            "validate_product_data",
            "Validate text against copy guidelines.",
            ValidateIn,
            P.CONTENT_GENERATE,
            "read",
            validate_product_data,
        ),
        ToolSpec(
            "prepare_product_update",
            "Create a PENDING change proposal that a human must approve.",
            PrepareUpdateIn,
            P.UPDATES_REQUEST,
            "write",
            prepare_product_update,
            requires_approval=True,
        ),
        ToolSpec(
            "approve_product_update",
            "Approve a pending proposal (admin, not the requester).",
            RequestIdIn,
            P.UPDATES_APPROVE,
            "write",
            approve_product_update,
        ),
        ToolSpec(
            "publish_product_update",
            "Publish an approved proposal to the product database.",
            RequestIdIn,
            P.UPDATES_APPROVE,
            "write",
            publish_product_update,
        ),
        ToolSpec(
            "run_analytics_query",
            "Admin-only read-only analytics SQL with AST validation and limits.",
            AnalyticsIn,
            P.ANALYTICS_SQL,
            "read",
            analytics_query,
        ),
    ]:
        r.register(spec)
    return r


_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = build_registry()
    return _registry


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
