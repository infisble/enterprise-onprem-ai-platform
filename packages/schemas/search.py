from __future__ import annotations

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Structured filters. Either extracted from natural language or passed explicitly."""

    price_min: float | None = None
    price_max: float | None = None
    category: str | None = None
    color: str | None = None
    material: str | None = None
    capacity_min_ml: int | None = None
    capacity_max_ml: int | None = None
    size: str | None = None
    in_stock: bool | None = None
    supplier: str | None = None
    brand: str | None = None
    language: str | None = None  # sk | cs | en (query language hint)
    sku: str | None = None

    def active(self) -> dict[str, object]:
        return {k: v for k, v in self.model_dump().items() if v not in (None, "")}


class ExtractedQuery(BaseModel):
    original: str
    semantic_text: str  # query with structured constraints stripped, used for retrieval
    filters: SearchFilters
    detected_language: str = "en"
    exact_code: str | None = None
    notes: list[str] = Field(default_factory=list)


class RankExplanation(BaseModel):
    bm25_rank: int | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    business_adjustment: float = 0.0
    matched_attributes: list[str] = Field(default_factory=list)


class ProductHit(BaseModel):
    product_id: int
    sku: str
    name: str
    name_en: str
    category: str
    color: str
    material: str
    capacity_ml: int | None
    sizes: list[str]
    price_eur: float
    stock_available: int | None
    supplier: str
    brand: str
    lead_time_days: int
    tags: list[str]
    final_score: float
    explain: RankExplanation


class SearchResponse(BaseModel):
    query: ExtractedQuery
    results: list[ProductHit]
    total_candidates: int
    stages: dict[str, dict]  # per-stage counts + latency for the demo UI
    relaxed_filters: list[str] = Field(default_factory=list)
    lexical_backend: str
    vector_backend: str
    reranker: str
    latency_ms: float
