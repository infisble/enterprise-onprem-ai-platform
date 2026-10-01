from sqlalchemy import select

from packages.core.db import session_scope
from packages.core.models import Product, ProductUpdateRequest
from services.agent import approvals
from services.agent.service import run_chat
from services.rag.service import get_rag_service


def test_rag_answer_has_page_level_citations(sales):
    res = get_rag_service().answer("What is our policy for urgent supplier deliveries?", sales)
    assert res.citations
    policy = [c for c in res.citations if c.source == "company_policy.pdf"]
    assert policy and policy[0].locator["page"] == 4
    assert "Purchasing Manager" in res.answer
    assert any(c.source == "supplier_terms.pdf" for c in res.citations), "sales may read supplier terms"


def test_chat_rag_route(sales):
    r = run_chat("What is our policy for urgent supplier deliveries?", sales)
    assert r.route == "rag" and r.citations
    assert r.llm["provider"] == "mock"
    assert r.latency["total_ms"] > 0


def test_stock_scenario_uses_typed_tool(sales):
    r = run_chat("How many pieces of SKU NP-10042 are currently available?", sales)
    assert r.route == "stock"
    assert [t.tool for t in r.tool_calls] == ["get_product_stock"]
    assert "1620" in r.answer


def test_content_agent_creates_pending_request_without_writing(catalog):
    with session_scope() as s:
        before = s.scalar(select(Product).where(Product.sku == "NP-10042"))
        before_desc, before_version = before.description_sk, before.version
    r = run_chat("Generate a new Slovak description for NP-10042.", catalog)
    assert r.route == "content"
    assert [t.tool for t in r.tool_calls] == [
        "get_product",
        "search_company_documents",
        "generate_product_description",
        "validate_product_data",
        "prepare_product_update",
    ]
    assert r.pending_approval and r.pending_approval["status"] == "pending"
    with session_scope() as s:
        after = s.scalar(select(Product).where(Product.sku == "NP-10042"))
        assert (after.description_sk, after.version) == (before_desc, before_version), (
            "no write before approval"
        )


def test_full_approval_flow(catalog, admin):
    r = run_chat("Generate a new Slovak description for NP-10042.", catalog)
    req_id = r.pending_approval["request_id"]
    # requester cannot approve (and does not have the permission anyway)
    try:
        approvals.approve(req_id, catalog)
        raise AssertionError("catalog manager must not approve")
    except Exception as exc:
        assert "updates:approve" in str(exc)
    approvals.approve(req_id, admin, "ok")
    published = approvals.publish(req_id, admin)
    assert published.status == "published"
    with session_scope() as s:
        p = s.scalar(select(Product).where(Product.sku == "NP-10042"))
        req = s.get(ProductUpdateRequest, req_id)
        assert p.description_sk == req.new_value
        assert p.version == req.base_version + 1
        assert p.content_source == "ai"


def test_stale_proposal_conflicts_and_tampering_is_detected(catalog, admin):
    with session_scope() as s:
        p = s.scalar(select(Product).where(Product.sku == "NP-10100"))
        sku = p.sku
    req_a = approvals.create_request(catalog, sku, "description_cs", "Varianta A " * 40, "test", {})
    req_b = approvals.create_request(catalog, sku, "description_cs", "Varianta B " * 40, "test", {})
    approvals.approve(req_a.id, admin)
    assert approvals.publish(req_a.id, admin).status == "published"
    approvals.approve(req_b.id, admin)
    assert approvals.publish(req_b.id, admin).status == "conflict"

    req_c = approvals.create_request(catalog, sku, "description_cs", "Varianta C " * 40, "test", {})
    approvals.approve(req_c.id, admin)
    with session_scope() as s:
        s.get(ProductUpdateRequest, req_c.id).new_value = "tampered after approval"
    try:
        approvals.publish(req_c.id, admin)
        raise AssertionError("tampering must be detected")
    except approvals.ApprovalError as exc:
        assert "hash mismatch" in str(exc)


def test_duplicate_proposals_are_idempotent(catalog):
    a = approvals.create_request(catalog, "NP-10200", "description_en", "Same text " * 40, "x", {})
    b = approvals.create_request(catalog, "NP-10200", "description_en", "Same text " * 40, "x", {})
    assert a.id == b.id


def test_session_state_enables_follow_up(sales):
    run_chat("Tell me about NP-10042", sales, conversation_id="c1")
    r = run_chat("How many of it are in stock?", sales, conversation_id="c1")
    assert r.route == "stock" and "NP-10042" in r.answer
