"""Structured filter extraction from natural-language queries (SK / CS / EN).

"Find black metal bottles around 750 ml under 15 EUR that are in stock" becomes
    color=black, material=metal, category=drinkware, capacity 675-825 ml, price_max=15, in_stock=True
and the residual semantic text "black metal bottles".

Deterministic rules are used first: they are fast, testable and never hallucinate a filter. With a real
LLM configured, an LLM JSON extractor can be layered on top for long-tail phrasing (see
`llm_refine_filters`), but its output is validated against the same Pydantic schema.
"""

from __future__ import annotations

import re

from packages.core.text import (
    COLOR_LOOKUP,
    MATERIAL_SYNONYMS,
    detect_language,
    normalize,
    product_type,
    tokenize,
)
from packages.schemas.search import ExtractedQuery, SearchFilters

_NUM = r"(\d+(?:[.,]\d+)?)"
_CUR = r"(?:\s*(?:eur|euro|eura|€|,-))"
_NOT_UNIT = r"(?!\s*(?:ml|l\b|mah|cm|mm|ks|pcs|g\b|w\b))"

PRICE_BETWEEN = re.compile(rf"(?:between|od|medzi|mezi)\s*{_NUM}\s*(?:and|a|do|-|–)\s*{_NUM}{_CUR}", re.I)
PRICE_RANGE = re.compile(rf"\b{_NUM}\s*(?:-|–)\s*{_NUM}{_CUR}", re.I)
PRICE_MAX = re.compile(
    rf"(?:under|below|less than|cheaper than|max(?:imum)?\.?|up to|at most|do|pod|menej ako|mene nez|"
    rf"lacnejsie ako|<=?)\s*(?:€\s*)?{_NUM}{_NOT_UNIT}{_CUR}?",
    re.I,
)
PRICE_MIN = re.compile(
    rf"(?:over|above|more than|at least|min(?:imum)?\.?|nad|viac ako|vice nez|>=?)\s*(?:€\s*)?{_NUM}{_NOT_UNIT}{_CUR}?",
    re.I,
)
PRICE_AROUND = re.compile(
    rf"(?:around|about|approximately|approx\.?|cca\.?|okolo|priblizne|asi)\s*(?:€\s*)?{_NUM}{_CUR}", re.I
)
CAPACITY = re.compile(
    rf"(?:(around|about|approximately|approx\.?|cca\.?|okolo|priblizne|asi)\s*)?{_NUM}\s*"
    r"(ml|l|liter|litre|liters|litres|litrov|litru|litry)\b",
    re.I,
)
SIZE_EXPLICIT = re.compile(r"\b(?:size|velkost|velikost|vel\.)\s*(xxs|xs|s|m|l|xl|xxl|3xl)\b", re.I)
SIZE_STANDALONE = re.compile(r"\b(xs|xl|xxl|3xl)\b", re.I)
IN_STOCK = re.compile(
    r"\b(?:in stock|on stock|available now|that are available|skladom|na sklade|skladem|na sklade|dostupne|dostupn[ey])\b",
    re.I,
)
SKU_CODES = [
    re.compile(r"\bNP-\d{5}\b", re.I),
    re.compile(r"\b\d{5}\.\d{2}\b"),
    re.compile(r"\bPT-\d{4}-[A-Z]{2}\b", re.I),
    re.compile(r"\bECO-\d{4}-\d\b", re.I),
]
SUPPLIERS = {"eurogift": "EGT", "promotex": "PTX", "ecoline": "ECO"}
BRANDS = {
    "thermaline",
    "novaline",
    "activego",
    "greenwave",
    "rainpro",
    "urbanpack",
    "scriba",
    "voltedge",
    "texline",
    "organico",
}


def _num(value: str) -> float:
    return float(value.replace(",", "."))


def extract_query(text: str) -> ExtractedQuery:
    original = text.strip()
    norm = normalize(original)
    filters = SearchFilters(language=detect_language(original))
    notes: list[str] = []
    spans: list[tuple[int, int]] = []

    exact = None
    for pattern in SKU_CODES:
        m = pattern.search(original)
        if m:
            exact = m.group(0).upper() if not m.group(0)[0].isdigit() else m.group(0)
            notes.append(f"exact product code detected: {exact}")
            break

    if m := PRICE_BETWEEN.search(norm) or PRICE_RANGE.search(norm):
        filters.price_min, filters.price_max = sorted((_num(m.group(1)), _num(m.group(2))))
        spans.append(m.span())
    else:
        if m := PRICE_AROUND.search(norm):
            v = _num(m.group(1))
            filters.price_min, filters.price_max = round(v * 0.8, 2), round(v * 1.2, 2)
            spans.append(m.span())
        if m := PRICE_MAX.search(norm):
            filters.price_max = _num(m.group(1))
            spans.append(m.span())
        if m := PRICE_MIN.search(norm):
            filters.price_min = _num(m.group(1))
            spans.append(m.span())

    if m := CAPACITY.search(norm):
        value = _num(m.group(2))
        ml = value * 1000 if m.group(3).startswith("l") else value
        tol = 0.15 if m.group(1) else 0.10
        filters.capacity_min_ml, filters.capacity_max_ml = int(ml * (1 - tol)), int(ml * (1 + tol))
        spans.append(m.span())

    if m := SIZE_EXPLICIT.search(norm) or SIZE_STANDALONE.search(norm):
        filters.size = m.group(1).upper()
        spans.append(m.span())

    if m := IN_STOCK.search(norm):
        filters.in_stock = True
        spans.append(m.span())

    for tok in tokenize(original):
        if filters.color is None and tok in COLOR_LOOKUP:
            filters.color = COLOR_LOOKUP[tok]
        if filters.material is None and tok in MATERIAL_SYNONYMS:
            filters.material = MATERIAL_SYNONYMS[tok]
        if filters.category is None and (ptype := product_type(tok)):
            filters.category = ptype[0]
        if filters.supplier is None and tok in SUPPLIERS:
            filters.supplier = SUPPLIERS[tok]
        if filters.brand is None and tok in BRANDS:
            filters.brand = tok

    # Residual semantic text: remove price / stock / size phrases; keep descriptive words.
    residual = norm
    for start, end in sorted(spans, reverse=True):
        residual = residual[:start] + " " + residual[end:]
    residual = re.sub(
        r"\b(find|show|search|looking for|i need|we need|that are|which are|hladam|potrebujem|"
        r"hledam|potrebuji|najdi|ukaz)\b",
        " ",
        residual,
    )
    residual = re.sub(r"\s+", " ", residual).strip(" ,.?!")
    if exact:
        residual = residual.replace(normalize(exact), "").strip()
    if len(tokenize(residual)) == 0:
        residual = norm

    return ExtractedQuery(
        original=original,
        semantic_text=residual,
        filters=filters,
        detected_language=filters.language or "en",
        exact_code=exact,
        notes=notes,
    )
