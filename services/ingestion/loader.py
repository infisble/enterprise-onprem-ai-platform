"""Load normalised supplier products into SQL (the source of truth) idempotently.

* Upsert key: (supplier, supplier_sku). Re-running a feed never duplicates products.
* SKU assignment: ERP crosswalk first; unmapped products get a newly allocated NP-2xxxx SKU.
* Content ownership: supplier-owned fields (price, stock, attributes) are refreshed from the feed,
  but descriptions edited by humans or approved AI content are never overwritten by a feed reload.
"""

from __future__ import annotations

import csv
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from packages.core.config import get_settings
from packages.core.models import (
    Customer,
    Order,
    OrderItem,
    PriceTier,
    Product,
    SkuCrosswalk,
    Stock,
    Supplier,
    User,
)
from packages.core.security import DEMO_USERS, hash_password
from packages.schemas.products import NormalizedProduct
from services.ingestion.suppliers import ADAPTERS, IngestionReport

SUPPLIER_META = {
    "EGT": {
        "default_lead_time_days": 7,
        "contact_email": "orders@eurogift.example",
        "urgent_delivery_supported": True,
        "notes": "Express 48h delivery available for orders placed before 11:00 CET (surcharge 6 %).",
    },
    "PTX": {
        "default_lead_time_days": 5,
        "contact_email": "b2b@promotex.example",
        "urgent_delivery_supported": True,
        "notes": "Textile printing capacity reserved for NovaPromo: 2,000 pcs/day.",
    },
    "ECO": {
        "default_lead_time_days": 12,
        "contact_email": "sales@ecoline.example",
        "urgent_delivery_supported": False,
        "notes": "No express option; minimum order 50 pcs; prices in CZK.",
    },
}
SUPPLIER_OWNED = (
    "name_en",
    "category",
    "subcategory",
    "color",
    "material",
    "capacity_ml",
    "sizes",
    "price_eur",
    "brand",
    "country_of_origin",
    "lead_time_days",
    "tags",
    "status",
)
CONTENT_FIELDS = ("name_sk", "name_cs", "description_sk", "description_en", "description_cs")


def ensure_suppliers(session: Session) -> dict[str, Supplier]:
    out = {}
    for a in ADAPTERS:
        sup = session.scalar(select(Supplier).where(Supplier.code == a.code))
        if sup is None:
            sup = Supplier(
                code=a.code,
                name=a.name,
                country=a.country,
                feed_format=a.feed_format,
                **SUPPLIER_META[a.code],
            )
            session.add(sup)
        out[a.code] = sup
    session.flush()
    return out


def load_crosswalk(session: Session, path: Path) -> None:
    if session.scalar(select(func.count()).select_from(SkuCrosswalk)):
        return
    with open(path, encoding="utf-8") as f:
        session.add_all(SkuCrosswalk(**row) for row in csv.DictReader(f))
    session.flush()


def _allocate_sku(session: Session) -> str:
    last = session.scalar(select(func.max(SkuCrosswalk.sku)).where(SkuCrosswalk.sku.like("NP-2%")))
    return f"NP-{int(last[3:]) + 1 if last else 20001}"


def upsert_products(session: Session, report: IngestionReport, data_dir: Path) -> dict[str, int]:
    suppliers = ensure_suppliers(session)
    load_crosswalk(session, data_dir / "products" / "sku_crosswalk.csv")
    xwalk = {(x.supplier_code, x.supplier_sku): x.sku for x in session.scalars(select(SkuCrosswalk))}
    own_stock: dict[str, dict[str, int]] = {}
    stock_file = data_dir / "products" / "warehouse_stock.csv"
    if stock_file.exists():
        with open(stock_file, encoding="utf-8") as f:
            own_stock = {r["sku"]: {k: int(v) for k, v in r.items() if k != "sku"} for r in csv.DictReader(f)}

    existing = {(p.supplier_id, p.supplier_sku): p for p in session.scalars(select(Product))}
    stats = {"inserted": 0, "updated": 0, "unchanged": 0, "new_skus": 0}
    for np_ in report.products:
        sup = suppliers[np_.supplier_code]
        product = existing.get((sup.id, np_.supplier_sku))
        if product is None:
            sku = xwalk.get((np_.supplier_code, np_.supplier_sku))
            if sku is None:
                sku = _allocate_sku(session)
                session.add(
                    SkuCrosswalk(supplier_code=np_.supplier_code, supplier_sku=np_.supplier_sku, sku=sku)
                )
                session.flush()
                stats["new_skus"] += 1
            product = _new_product(np_, sup.id, sku)
            session.add(product)
            session.flush()
            _write_stock_and_prices(session, product, np_, own_stock.get(sku, {}))
            stats["inserted"] += 1
            continue
        changed = False
        for field in SUPPLIER_OWNED:
            new = getattr(np_, field)
            if getattr(product, field) != new:
                setattr(product, field, new)
                changed = True
        if product.content_source == "supplier":
            for field in CONTENT_FIELDS:
                new = getattr(np_, field) or (np_.name_sk if field == "name_sk" else "")
                if new and getattr(product, field) != new:
                    setattr(product, field, new)
                    changed = True
        _write_stock_and_prices(session, product, np_, None)
        if changed:
            product.version += 1
            product.updated_at = datetime.now(UTC)
            stats["updated"] += 1
        else:
            stats["unchanged"] += 1
    return stats


def _new_product(np_: NormalizedProduct, supplier_id: int, sku: str) -> Product:
    return Product(
        sku=sku,
        supplier_id=supplier_id,
        supplier_sku=np_.supplier_sku,
        name_sk=np_.name_sk or np_.name_en,
        name_en=np_.name_en,
        name_cs=np_.name_cs,
        description_sk=np_.description_sk,
        description_en=np_.description_en,
        description_cs=np_.description_cs,
        category=np_.category,
        subcategory=np_.subcategory,
        color=np_.color,
        material=np_.material,
        capacity_ml=np_.capacity_ml,
        sizes=np_.sizes,
        price_eur=np_.price_eur,
        brand=np_.brand,
        country_of_origin=np_.country_of_origin,
        lead_time_days=np_.lead_time_days,
        tags=np_.tags,
        status=np_.status,
    )


def _write_stock_and_prices(
    session: Session, p: Product, np_: NormalizedProduct, own: dict[str, int] | None
) -> None:
    now = datetime.now(UTC)
    rows = {s.warehouse: s for s in session.scalars(select(Stock).where(Stock.product_id == p.id))}
    targets = {"SUPPLIER": (np_.stock, 0)}
    if own is not None:
        targets["BA"] = (own.get("BA", 0), own.get("reserved", 0))
        targets["ZA"] = (own.get("ZA", 0), 0)
    for wh, (qty, reserved) in targets.items():
        row = rows.get(wh)
        if row is None:
            session.add(Stock(product_id=p.id, warehouse=wh, quantity=qty, reserved=reserved, updated_at=now))
        elif row.quantity != qty:
            row.quantity, row.updated_at = qty, now
    if not session.scalar(select(func.count()).select_from(PriceTier).where(PriceTier.product_id == p.id)):
        for qty, factor in ((1, 1.0), (100, 0.92), (500, 0.85), (2000, 0.8)):
            session.add(
                PriceTier(product_id=p.id, min_qty=qty, unit_price_eur=round(p.price_eur * factor, 2))
            )


def seed_users(session: Session) -> int:
    pwd = get_settings().demo_password
    n = 0
    for u in DEMO_USERS:
        if not session.scalar(select(User).where(User.username == u["username"])):
            session.add(
                User(
                    username=u["username"],
                    full_name=u["full_name"],
                    role=u["role"],
                    password_hash=hash_password(pwd),
                )
            )
            n += 1
    return n


def seed_customers_and_orders(session: Session, n_customers: int = 120, n_orders: int = 900) -> int:
    if session.scalar(select(func.count()).select_from(Customer)):
        return 0
    rng = random.Random(7)
    companies = [
        "Tatra",
        "Danubia",
        "Morava",
        "Vltava",
        "Carpathia",
        "Nitra",
        "Orava",
        "Krkonoše",
        "Považie",
        "Šumava",
        "Spiš",
        "Záhorie",
        "Liptov",
        "Hradec",
        "Brno",
        "Košice",
    ]
    kinds = [
        "Software s.r.o.",
        "Logistics a.s.",
        "Pharma s.r.o.",
        "Bank a.s.",
        "Consulting s.r.o.",
        "Energy a.s.",
        "Events s.r.o.",
        "Retail a.s.",
        "Automotive s.r.o.",
        "Telecom a.s.",
    ]
    first = ["Ján", "Petra", "Martin", "Zuzana", "Tomáš", "Lucia", "Jakub", "Veronika", "Michal", "Tereza"]
    last = [
        "Novák",
        "Kováčová",
        "Horváth",
        "Svoboda",
        "Varga",
        "Dvořáková",
        "Tóth",
        "Černý",
        "Baláž",
        "Procházka",
    ]
    customers = []
    for i in range(n_customers):
        company = f"{rng.choice(companies)} {rng.choice(kinds)}"
        name = f"{rng.choice(first)} {rng.choice(last)}"
        customers.append(
            Customer(
                company=company,
                contact_name=name,
                email=f"contact{i}@{company.split()[0].lower()}-client.example",
                country=rng.choice(["SK", "SK", "CZ"]),
            )
        )
    session.add_all(customers)
    session.flush()
    product_ids = [pid for (pid,) in session.execute(select(Product.id))]
    prices = dict(session.execute(select(Product.id, Product.price_eur)).all())
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for _ in range(n_orders):
        order = Order(
            customer_id=rng.choice(customers).id,
            status=rng.choice(["delivered"] * 6 + ["open", "cancelled"]),
            created_at=start + timedelta(days=rng.randint(0, 270), hours=rng.randint(7, 18)),
            total_eur=0,
        )
        session.add(order)
        session.flush()
        total = 0.0
        for pid in rng.sample(product_ids, rng.randint(1, 4)):
            qty = rng.choice([50, 100, 150, 250, 500, 1000])
            unit = round(prices[pid] * (0.92 if qty >= 100 else 1.0), 2)
            session.add(OrderItem(order_id=order.id, product_id=pid, quantity=qty, unit_price_eur=unit))
            total += qty * unit
        order.total_eur = round(total, 2)
    return n_customers
