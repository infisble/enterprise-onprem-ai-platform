"""Product-oriented search documents.

A product is indexed as ONE document (not arbitrary 500-token slices): names in all languages,
descriptions, attributes and tags. The same document feeds Elasticsearch (BM25 + filters) and
Qdrant (embedding + payload filters). SQL stays the source of truth; both indexes are rebuildable.
"""

from __future__ import annotations

from typing import Any

from packages.core.models import Product


def product_search_doc(p: Product, stock_available: int, supplier_code: str) -> dict[str, Any]:
    return {
        "product_id": p.id,
        "sku": p.sku,
        "supplier_sku": p.supplier_sku,
        "supplier": supplier_code,
        "name_sk": p.name_sk,
        "name_en": p.name_en,
        "name_cs": p.name_cs,
        "description_sk": p.description_sk,
        "description_en": p.description_en,
        "description_cs": p.description_cs,
        "category": p.category,
        "subcategory": p.subcategory,
        "color": p.color,
        "material": p.material,
        "capacity_ml": p.capacity_ml,
        "sizes": p.sizes or [],
        "price_eur": p.price_eur,
        "stock_available": stock_available,
        "brand": (p.brand or "").lower(),
        "lead_time_days": p.lead_time_days,
        "tags": p.tags or [],
        "status": p.status,
    }


def embedding_text(doc: dict[str, Any]) -> str:
    """Text used for the product's semantic vector: multilingual names + attributes + tags + description."""
    attrs = [doc["category"], doc["subcategory"], doc["material"], doc["color"]]
    if doc.get("capacity_ml"):
        attrs.append(f"{doc['capacity_ml']} ml")
    return " . ".join(
        filter(
            None,
            [
                doc["name_en"],
                doc["name_sk"],
                doc.get("name_cs", ""),
                " ".join(attrs),
                " ".join(doc.get("tags", [])),
                doc.get("description_en", ""),
            ],
        )
    )


def lexical_text_fields(doc: dict[str, Any]) -> dict[str, float]:
    """Fields and boosts used by the in-process BM25 fallback (mirrors the ES multi_match)."""
    return {
        doc["name_en"]: 3.0,
        doc["name_sk"]: 3.0,
        doc.get("name_cs", ""): 2.0,
        " ".join(doc.get("tags", [])): 2.0,
        f"{doc['category']} {doc['subcategory']}": 1.5,
        f"{doc['material']} {doc['color']} {doc.get('brand', '')}": 1.5,
        doc.get("description_en", ""): 1.0,
        doc.get("description_sk", ""): 1.0,
    }
