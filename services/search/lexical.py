"""Lexical retrieval: BM25, exact codes, structured filters, facets.

Production backend is Elasticsearch 8 (ADR-002): mature BM25, ASCII folding for Slovak/Czech
diacritics, edge-ngrams for partial keyword search, keyword fields for SKUs, range filters and
aggregations for facets. `InMemoryBM25Index` implements the same interface for the laptop demo and
unit tests, so behaviour can be compared directly.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Protocol

from packages.core.config import get_settings
from packages.core.text import MATERIAL_FAMILIES, stem, tokenize
from packages.observability.logging import get_logger
from packages.schemas.search import SearchFilters
from services.search.documents import lexical_text_fields

log = get_logger(__name__)


class LexicalIndex(Protocol):
    backend: str

    def index(self, docs: list[dict[str, Any]], recreate: bool = False) -> int: ...
    def search(self, query: str, filters: SearchFilters, k: int) -> list[tuple[int, float]]: ...
    def find_code(self, code: str) -> list[int]: ...
    def facets(self, filters: SearchFilters) -> dict[str, dict[str, int]]: ...
    def count(self) -> int: ...
    def ping(self) -> bool: ...


def material_values(material: str | None) -> list[str] | None:
    if not material:
        return None
    return MATERIAL_FAMILIES.get(material, [material])


def matches_filters(doc: dict[str, Any], f: SearchFilters) -> bool:
    if doc.get("status") != "active":
        return False
    if f.price_min is not None and doc["price_eur"] < f.price_min:
        return False
    if f.price_max is not None and doc["price_eur"] > f.price_max:
        return False
    if f.category and doc["category"] != f.category:
        return False
    if f.color and doc["color"] != f.color:
        return False
    mats = material_values(f.material)
    if mats and doc["material"] not in mats:
        return False
    cap = doc.get("capacity_ml")
    if f.capacity_min_ml is not None and (cap is None or cap < f.capacity_min_ml):
        return False
    if f.capacity_max_ml is not None and (cap is None or cap > f.capacity_max_ml):
        return False
    if f.size and f.size not in (doc.get("sizes") or []):
        return False
    if f.in_stock and doc.get("stock_available", 0) <= 0:
        return False
    if f.supplier and doc["supplier"] != f.supplier:
        return False
    return not (f.brand and doc.get("brand") != f.brand.lower())


class InMemoryBM25Index:
    backend = "in-process-bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: dict[int, dict[str, Any]] = {}
        self._tf: dict[int, Counter[str]] = {}
        self._len: dict[int, float] = {}
        self._df: Counter[str] = Counter()
        self._avg = 1.0
        self._codes: dict[str, list[int]] = defaultdict(list)

    def index(self, docs: list[dict[str, Any]], recreate: bool = False) -> int:
        if recreate:
            self.__init__(self.k1, self.b)  # type: ignore[misc]
        for d in docs:
            pid = d["product_id"]
            if pid in self._tf:
                for term in self._tf[pid]:
                    self._df[term] -= 1
            tf: Counter[str] = Counter()  # values are boosted (float) term frequencies
            for text, boost in lexical_text_fields(d).items():
                for tok in tokenize(text):
                    tf[stem(tok)] += boost  # type: ignore[assignment]
            self.docs[pid] = d
            self._tf[pid] = tf
            self._len[pid] = sum(tf.values())
            for term in tf:
                self._df[term] += 1
            for code in (d["sku"], d["supplier_sku"]):
                if pid not in self._codes[code.upper()]:
                    self._codes[code.upper()].append(pid)
        self._avg = sum(self._len.values()) / max(1, len(self._len))
        return len(docs)

    def search(self, query: str, filters: SearchFilters, k: int) -> list[tuple[int, float]]:
        terms = [stem(t) for t in tokenize(query)]
        n = len(self.docs)
        scores: dict[int, float] = {}
        for pid, tf in self._tf.items():
            if not matches_filters(self.docs[pid], filters):
                continue
            s = 0.0
            for term in terms:
                f = tf.get(term)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                s += (
                    idf
                    * f
                    * (self.k1 + 1)
                    / (f + self.k1 * (1 - self.b + self.b * self._len[pid] / self._avg))
                )
            if s > 0:
                scores[pid] = s
        return sorted(scores.items(), key=lambda x: -x[1])[:k]

    def find_code(self, code: str) -> list[int]:
        return list(self._codes.get(code.upper(), []))

    def facets(self, filters: SearchFilters) -> dict[str, dict[str, int]]:
        out: dict[str, Counter[str]] = {
            "category": Counter(),
            "color": Counter(),
            "material": Counter(),
            "supplier": Counter(),
        }
        for d in self.docs.values():
            if matches_filters(d, filters):
                for key in out:
                    out[key][d[key]] += 1
        return {k: dict(v.most_common(12)) for k, v in out.items()}

    def count(self) -> int:
        return len(self.docs)

    def ping(self) -> bool:
        return True


ES_SETTINGS = {
    "analysis": {
        "filter": {"edge": {"type": "edge_ngram", "min_gram": 2, "max_gram": 15}},
        "analyzer": {
            # Slovak/Czech: lowercase + ASCII folding so "fľaša" matches "flasa" and vice versa.
            "folded": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "asciifolding"]},
            "folded_prefix": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "asciifolding", "edge"],
            },
            "english_folded": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "asciifolding", "porter_stem"],
            },
        },
    },
    "number_of_shards": 1,
    "number_of_replicas": 0,
}
_TEXT = {
    "type": "text",
    "analyzer": "folded",
    "fields": {"prefix": {"type": "text", "analyzer": "folded_prefix", "search_analyzer": "folded"}},
}
ES_MAPPINGS = {
    "properties": {
        "product_id": {"type": "integer"},
        "sku": {"type": "keyword"},
        "supplier_sku": {"type": "keyword"},
        "supplier": {"type": "keyword"},
        "name_sk": _TEXT,
        "name_cs": _TEXT,
        "name_en": {
            "type": "text",
            "analyzer": "english_folded",
            "fields": {"prefix": {"type": "text", "analyzer": "folded_prefix", "search_analyzer": "folded"}},
        },
        "description_sk": {"type": "text", "analyzer": "folded"},
        "description_cs": {"type": "text", "analyzer": "folded"},
        "description_en": {"type": "text", "analyzer": "english_folded"},
        "category": {"type": "keyword"},
        "subcategory": {"type": "keyword"},
        "color": {"type": "keyword"},
        "material": {"type": "keyword"},
        "capacity_ml": {"type": "integer"},
        "sizes": {"type": "keyword"},
        "price_eur": {"type": "float"},
        "stock_available": {"type": "integer"},
        "brand": {"type": "keyword"},
        "lead_time_days": {"type": "integer"},
        "tags": {"type": "text", "analyzer": "folded", "fields": {"kw": {"type": "keyword"}}},
        "status": {"type": "keyword"},
    }
}


def es_filter_clauses(f: SearchFilters) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = [{"term": {"status": "active"}}]
    price: dict[str, float] = {}
    if f.price_min is not None:
        price["gte"] = f.price_min
    if f.price_max is not None:
        price["lte"] = f.price_max
    if price:
        clauses.append({"range": {"price_eur": price}})
    cap: dict[str, int] = {}
    if f.capacity_min_ml is not None:
        cap["gte"] = f.capacity_min_ml
    if f.capacity_max_ml is not None:
        cap["lte"] = f.capacity_max_ml
    if cap:
        clauses.append({"range": {"capacity_ml": cap}})
    for field, value in (
        ("category", f.category),
        ("color", f.color),
        ("sizes", f.size),
        ("supplier", f.supplier),
        ("brand", f.brand.lower() if f.brand else None),
    ):
        if value:
            clauses.append({"term": {field: value}})
    if mats := material_values(f.material):
        clauses.append({"terms": {"material": mats}})
    if f.in_stock:
        clauses.append({"range": {"stock_available": {"gt": 0}}})
    return clauses


class ElasticsearchLexicalIndex:
    backend = "elasticsearch"

    def __init__(self, url: str, index: str) -> None:
        from elasticsearch import Elasticsearch

        self.es = Elasticsearch(url, request_timeout=30)
        self.index_name = index

    def index(self, docs: list[dict[str, Any]], recreate: bool = False) -> int:
        from elasticsearch import helpers

        if recreate and self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, settings=ES_SETTINGS, mappings=ES_MAPPINGS)
        actions = ({"_index": self.index_name, "_id": d["product_id"], "_source": d} for d in docs)
        ok, _ = helpers.bulk(self.es, actions, refresh="wait_for")
        return int(ok)

    def search(self, query: str, filters: SearchFilters, k: int) -> list[tuple[int, float]]:
        body = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "name_en^3",
                                "name_sk^3",
                                "name_cs^2",
                                "name_en.prefix",
                                "name_sk.prefix",
                                "tags^2",
                                "description_en",
                                "description_sk",
                                "description_cs",
                                "category^1.5",
                                "subcategory^1.5",
                                "material^1.5",
                                "color^1.5",
                            ],
                            "type": "most_fields",
                            "fuzziness": "AUTO",
                            "prefix_length": 2,
                        }
                    }
                ],
                "filter": es_filter_clauses(filters),
            }
        }
        res = self.es.search(index=self.index_name, query=body, size=k, source=False)
        return [(int(h["_id"]), float(h["_score"])) for h in res["hits"]["hits"]]

    def find_code(self, code: str) -> list[int]:
        res = self.es.search(
            index=self.index_name,
            size=5,
            source=False,
            query={
                "bool": {
                    "should": [
                        {"term": {"sku": code.upper()}},
                        {"term": {"supplier_sku": code}},
                        {"term": {"supplier_sku": code.upper()}},
                    ]
                }
            },
        )
        return [int(h["_id"]) for h in res["hits"]["hits"]]

    def facets(self, filters: SearchFilters) -> dict[str, dict[str, int]]:
        aggs = {k: {"terms": {"field": k, "size": 12}} for k in ("category", "color", "material", "supplier")}
        res = self.es.search(
            index=self.index_name, size=0, aggs=aggs, query={"bool": {"filter": es_filter_clauses(filters)}}
        )
        return {k: {b["key"]: b["doc_count"] for b in v["buckets"]} for k, v in res["aggregations"].items()}

    def count(self) -> int:
        try:
            return int(self.es.count(index=self.index_name)["count"])
        except Exception:
            return 0

    def ping(self) -> bool:
        try:
            return bool(self.es.ping())
        except Exception:
            return False


_index: LexicalIndex | None = None


def get_lexical_index() -> LexicalIndex:
    global _index
    if _index is None:
        s = get_settings()
        _index = (
            ElasticsearchLexicalIndex(s.elasticsearch_url, s.products_index)
            if s.elasticsearch_url
            else InMemoryBM25Index()
        )
    return _index


def reset_lexical_index() -> None:
    global _index
    _index = None
