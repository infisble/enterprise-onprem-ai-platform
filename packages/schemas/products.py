from __future__ import annotations

from pydantic import BaseModel, Field


class NormalizedProduct(BaseModel):
    """Canonical product produced by supplier-feed normalisation, independent of feed schema."""

    supplier_code: str
    supplier_sku: str
    name_sk: str
    name_en: str
    name_cs: str = ""
    description_sk: str = ""
    description_en: str = ""
    description_cs: str = ""
    category: str
    subcategory: str = ""
    color: str
    material: str = ""
    capacity_ml: int | None = None
    sizes: list[str] = Field(default_factory=list)
    price_eur: float = Field(gt=0)
    stock: int = Field(ge=0, default=0)
    brand: str = ""
    country_of_origin: str = ""
    lead_time_days: int = Field(ge=0, default=10)
    tags: list[str] = Field(default_factory=list)
    status: str = "active"


class ProductOut(BaseModel):
    product_id: int
    sku: str
    supplier_sku: str
    supplier: str
    name_sk: str
    name_en: str
    name_cs: str
    description_sk: str
    description_en: str
    description_cs: str
    category: str
    subcategory: str
    color: str
    material: str
    capacity_ml: int | None
    sizes: list[str]
    price_eur: float
    brand: str
    country_of_origin: str
    lead_time_days: int
    tags: list[str]
    status: str
    content_source: str
    needs_review: bool
    version: int
    stock_available: int | None = None  # None => caller not allowed to see stock


class StockOut(BaseModel):
    sku: str
    total_available: int
    by_warehouse: dict[str, int]
    lead_time_days: int
    as_of: str


class PriceOut(BaseModel):
    sku: str
    list_price_eur: float
    tiers: list[dict[str, float]]
    currency: str = "EUR"
