from sqlalchemy import func, select

from packages.core.db import session_scope
from packages.core.models import Product, Stock
from services.agent.tools import get_registry
from services.search.hybrid import get_search_service


def test_hybrid_search_respects_every_structured_constraint(sales):
    res = get_search_service().search(
        "Find black metal bottles around 750 ml under 15 EUR that are in stock.", sales
    )
    assert res.results, "expected matches"
    for h in res.results:
        assert h.color == "black"
        assert h.material in ("stainless steel", "aluminium")
        assert h.price_eur <= 15
        assert 637 <= h.capacity_ml <= 863
        assert h.stock_available > 0
    assert "NP-10042" in [h.sku for h in res.results]
    assert {"bm25", "vector", "fusion", "rerank"} <= set(res.stages)
    top = res.results[0].explain
    assert top.rrf_score > 0 and top.rerank_score is not None


def test_semantic_discovery_returns_eco_products(employee):
    res = get_search_service().search("I need eco-friendly gifts for a technology conference.", employee)
    eco_materials = {
        "bamboo",
        "cork",
        "recycled pet",
        "recycled paper",
        "jute",
        "organic cotton",
        "wheat straw",
        "wood",
    }
    top = res.results[:6]
    assert sum(h.material in eco_materials or "eco" in h.tags for h in top) >= 5
    assert len({h.name_en.split()[0] for h in top}) >= 2, "results should be diversified"


def test_exact_code_short_circuits(employee):
    res = get_search_service().search("product 29904.01", employee)
    assert [h.sku for h in res.results] == ["NP-10042"]
    assert "exact_match" in res.stages and "vector" not in res.stages


def test_employee_does_not_see_stock_numbers(employee, sales):
    q = "black thermal bottle"
    assert all(h.stock_available is None for h in get_search_service().search(q, employee).results)
    assert all(h.stock_available is not None for h in get_search_service().search(q, sales).results)


def test_lexical_vs_vector_modes_differ(sales):
    svc = get_search_service()
    lex = svc.search("eco gifts", sales, mode="lexical")
    vec = svc.search("eco gifts", sales, mode="vector")
    assert "vector" not in lex.stages and "bm25" not in vec.stages


def test_relaxation_of_soft_filters(sales):
    res = get_search_service().search("bamboo umbrella", sales)
    assert res.relaxed_filters, "no bamboo umbrellas exist, soft filters should be relaxed"


def test_stock_tool_matches_sql_source_of_truth(sales):
    result, rec = get_registry().execute("get_product_stock", {"sku": "NP-10042"}, sales)
    assert rec.outcome == "success"
    with session_scope() as s:
        pid = s.scalar(select(Product.id).where(Product.sku == "NP-10042"))
        expected = s.scalar(select(func.sum(Stock.quantity - Stock.reserved)).where(Stock.product_id == pid))
    assert result["total_available"] == expected == 1620
    assert result["by_warehouse"] == {"SUPPLIER": 1200, "BA": 300, "ZA": 120}


def test_price_tool_applies_volume_tiers(sales):
    result, rec = get_registry().execute("get_product_price", {"sku": "NP-10042", "quantity": 500}, sales)
    assert rec.outcome == "success"
    assert result["list_price_eur"] == 12.90
    assert result["unit_price_eur"] == round(12.90 * 0.85, 2)
    assert result["total_eur"] == round(result["unit_price_eur"] * 500, 2)


def test_tool_argument_validation(sales):
    _, rec = get_registry().execute("get_product_stock", {"sku": "NP-10042; DROP TABLE products"}, sales)
    assert rec.outcome == "invalid_args"
    _, rec = get_registry().execute("get_product_price", {"sku": "NP-10042", "quantity": -5}, sales)
    assert rec.outcome == "invalid_args"
    _, rec = get_registry().execute("get_product_stock", {"sku": "NP-99999"}, sales)
    assert rec.outcome == "error" and "not found" in rec.error
