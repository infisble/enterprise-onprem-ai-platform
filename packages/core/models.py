"""Relational model. SQL remains the source of truth for everything transactional:
products, stock, prices, suppliers, customers, orders, users, approvals, audit and batch state.
Qdrant and Elasticsearch only hold *derived, rebuildable* projections of this data."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.core.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Supplier(Base):
    __tablename__ = "suppliers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    country: Mapped[str] = mapped_column(String(2))
    feed_format: Mapped[str] = mapped_column(String(16))
    default_lead_time_days: Mapped[int] = mapped_column(Integer, default=10)
    contact_email: Mapped[str] = mapped_column(String(200), default="")
    urgent_delivery_supported: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")


class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sku: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.id"), index=True)
    supplier_sku: Mapped[str] = mapped_column(String(64), index=True)
    name_sk: Mapped[str] = mapped_column(String(300))
    name_en: Mapped[str] = mapped_column(String(300))
    name_cs: Mapped[str] = mapped_column(String(300), default="")
    description_sk: Mapped[str] = mapped_column(Text, default="")
    description_en: Mapped[str] = mapped_column(Text, default="")
    description_cs: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(64), index=True)
    subcategory: Mapped[str] = mapped_column(String(64), default="")
    color: Mapped[str] = mapped_column(String(32), index=True)
    material: Mapped[str] = mapped_column(String(64), default="")
    capacity_ml: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sizes: Mapped[list[str]] = mapped_column(JSON, default=list)
    price_eur: Mapped[float] = mapped_column(Float)
    brand: Mapped[str] = mapped_column(String(64), default="")
    country_of_origin: Mapped[str] = mapped_column(String(2), default="")
    lead_time_days: Mapped[int] = mapped_column(Integer, default=10)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | discontinued
    content_source: Mapped[str] = mapped_column(String(16), default="supplier")  # supplier|human|ai
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    supplier: Mapped[Supplier] = relationship()
    stock_rows: Mapped[list[Stock]] = relationship(back_populates="product")
    price_tiers: Mapped[list[PriceTier]] = relationship(back_populates="product")


class Stock(Base):
    __tablename__ = "stock"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    warehouse: Mapped[str] = mapped_column(String(32))  # BA (Bratislava), ZA (Zilina), SUPPLIER
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    reserved: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    product: Mapped[Product] = relationship(back_populates="stock_rows")
    __table_args__ = (UniqueConstraint("product_id", "warehouse"),)


class PriceTier(Base):
    __tablename__ = "price_tiers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    min_qty: Mapped[int] = mapped_column(Integer)
    unit_price_eur: Mapped[float] = mapped_column(Float)
    product: Mapped[Product] = relationship(back_populates="price_tiers")


class SkuCrosswalk(Base):
    """ERP mapping supplier product code -> NovaPromo SKU (stable across feed reloads)."""

    __tablename__ = "sku_crosswalk"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    supplier_code: Mapped[str] = mapped_column(String(32))
    supplier_sku: Mapped[str] = mapped_column(String(64))
    sku: Mapped[str] = mapped_column(String(32), unique=True)
    __table_args__ = (UniqueConstraint("supplier_code", "supplier_sku"),)


class Customer(Base):
    """Restricted PII table. No agent tool exposes it; used to prove injection resistance."""

    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company: Mapped[str] = mapped_column(String(200))
    contact_name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(200))
    country: Mapped[str] = mapped_column(String(2))


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    status: Mapped[str] = mapped_column(String(16))
    total_eur: Mapped[float] = mapped_column(Float)


class OrderItem(Base):
    __tablename__ = "order_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price_eur: Mapped[float] = mapped_column(Float)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    full_name: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(32))
    password_hash: Mapped[str] = mapped_column(String(256))
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(300), unique=True)
    doc_type: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(300))
    access_roles: Mapped[list[str]] = mapped_column(JSON)
    classification: Mapped[str] = mapped_column(String(32), default="internal")
    sha256: Mapped[str] = mapped_column(String(64))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    flagged_chunks: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProductUpdateRequest(Base):
    """Human-in-the-loop write boundary. Agents can only *propose* changes here."""

    __tablename__ = "product_update_requests"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    sku: Mapped[str] = mapped_column(String(32))
    field: Mapped[str] = mapped_column(String(64))
    old_value: Mapped[str] = mapped_column(Text, default="")
    new_value: Mapped[str] = mapped_column(Text)
    payload_hash: Mapped[str] = mapped_column(String(64))
    base_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(
        String(16), default="pending"
    )  # pending|approved|rejected|published|conflict
    source: Mapped[str] = mapped_column(String(16), default="agent")  # agent | batch | human
    validation: Mapped[dict] = mapped_column(JSON, default=dict)
    rationale: Mapped[str] = mapped_column(Text, default="")
    request_id: Mapped[str] = mapped_column(String(64), default="")
    requested_by: Mapped[str] = mapped_column(String(64))
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decision_note: Mapped[str] = mapped_column(Text, default="")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    request_id: Mapped[str] = mapped_column(String(64), default="")
    username: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(64))
    resource: Mapped[str] = mapped_column(String(200), default="")
    outcome: Mapped[str] = mapped_column(String(16))  # allowed | denied | error | success
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class BatchJob(Base):
    __tablename__ = "batch_jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        String(16), default="queued"
    )  # queued|running|completed|failed|paused
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    created_by: Mapped[str] = mapped_column(String(64), default="scheduler")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total: Mapped[int] = mapped_column(Integer, default=0)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    succeeded: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    checkpoint_product_id: Mapped[int] = mapped_column(Integer, default=0)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")


class BatchItem(Base):
    __tablename__ = "batch_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("batch_jobs.id"), index=True)
    product_id: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|succeeded|failed|skipped
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str] = mapped_column(Text, default="")
    actions: Mapped[list[str]] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("job_id", "product_id"),)


class ProcessingLedger(Base):
    """Idempotency ledger: (product, task, input_hash) processed once, regardless of job."""

    __tablename__ = "processing_ledger"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(Integer, index=True)
    task: Mapped[str] = mapped_column(String(32))
    input_hash: Mapped[str] = mapped_column(String(64))
    job_id: Mapped[int] = mapped_column(Integer)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("product_id", "task", "input_hash"),)


class RequestTrace(Base):
    __tablename__ = "request_traces"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    username: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(32))
    kind: Mapped[str] = mapped_column(String(16), default="chat")  # chat | search | batch
    query: Mapped[str] = mapped_column(Text, default="")
    route: Mapped[str] = mapped_column(String(32), default="")
    llm_provider: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(128), default="")
    total_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    spans: Mapped[list[dict]] = mapped_column(JSON, default=list)
    tool_calls: Mapped[list[dict]] = mapped_column(JSON, default=list)
    error: Mapped[str] = mapped_column(Text, default="")
