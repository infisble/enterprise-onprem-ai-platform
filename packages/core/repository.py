"""Read helpers over the SQL source of truth (products, stock, prices)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from packages.core.models import PriceTier, Product, Stock, Supplier
from packages.schemas.products import PriceOut, ProductOut, StockOut


def available_stock(session: Session, product_ids: list[int]) -> dict[int, int]:
    if not product_ids:
        return {}
    rows = session.execute(
        select(Stock.product_id, func.sum(Stock.quantity - Stock.reserved))
        .where(Stock.product_id.in_(product_ids))
        .group_by(Stock.product_id)
    ).all()
    out = {pid: 0 for pid in product_ids}
    out.update({pid: max(0, int(total or 0)) for pid, total in rows})
    return out


def products_by_ids(session: Session, product_ids: list[int]) -> dict[int, Product]:
    if not product_ids:
        return {}
    rows = session.scalars(
        select(Product).options(joinedload(Product.supplier)).where(Product.id.in_(product_ids))
    )
    return {p.id: p for p in rows}


def product_by_sku(session: Session, sku: str) -> Product | None:
    sku = sku.strip()
    p = session.scalar(
        select(Product).options(joinedload(Product.supplier)).where(Product.sku == sku.upper())
    )
    if p is None:  # allow supplier codes too (e.g. 29904.01)
        p = session.scalar(
            select(Product).options(joinedload(Product.supplier)).where(Product.supplier_sku == sku)
        )
    return p


def product_out(p: Product, stock: int | None) -> ProductOut:
    return ProductOut(
        product_id=p.id,
        sku=p.sku,
        supplier_sku=p.supplier_sku,
        supplier=p.supplier.name if p.supplier else "",
        name_sk=p.name_sk,
        name_en=p.name_en,
        name_cs=p.name_cs,
        description_sk=p.description_sk,
        description_en=p.description_en,
        description_cs=p.description_cs,
        category=p.category,
        subcategory=p.subcategory,
        color=p.color,
        material=p.material,
        capacity_ml=p.capacity_ml,
        sizes=p.sizes or [],
        price_eur=p.price_eur,
        brand=p.brand,
        country_of_origin=p.country_of_origin,
        lead_time_days=p.lead_time_days,
        tags=p.tags or [],
        status=p.status,
        content_source=p.content_source,
        needs_review=p.needs_review,
        version=p.version,
        stock_available=stock,
    )


def stock_out(session: Session, p: Product) -> StockOut:
    rows = session.scalars(select(Stock).where(Stock.product_id == p.id)).all()
    by_wh = {r.warehouse: max(0, r.quantity - r.reserved) for r in rows}
    as_of = max((r.updated_at for r in rows), default=datetime.now(UTC))
    return StockOut(
        sku=p.sku,
        total_available=sum(by_wh.values()),
        by_warehouse=by_wh,
        lead_time_days=p.lead_time_days,
        as_of=as_of.isoformat(),
    )


def price_out(session: Session, p: Product) -> PriceOut:
    tiers = session.scalars(
        select(PriceTier).where(PriceTier.product_id == p.id).order_by(PriceTier.min_qty)
    ).all()
    return PriceOut(
        sku=p.sku,
        list_price_eur=p.price_eur,
        tiers=[{"min_qty": t.min_qty, "unit_price_eur": t.unit_price_eur} for t in tiers],
    )


def supplier_codes(session: Session) -> dict[int, str]:
    return {s.id: s.code for s in session.scalars(select(Supplier))}
