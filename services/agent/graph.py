"""LangGraph orchestration (ADR-005).

    guard ─► route ─┬─► search ──────────────┐
                    ├─► rag ─────────────────┤
                    ├─► stock / price / product_info / supplier ─┤
                    ├─► analytics ───────────┤
                    ├─► translate ───────────┤
                    ├─► refuse / general ────┤
                    └─► content_gather ─► content_generate ─► content_validate ─┬─► content_propose ─┤
                                              ▲                                 │ (invalid, retry<2)   │
                                              └─────────────────────────────────┘                      ▼
                                                                                                   finalize

The graph is explicit code: every edge is reviewable and testable. Nodes call tools only through the
governed `ToolRegistry.execute`, which enforces RBAC, validates arguments and audits.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from packages.core.security import Principal
from packages.observability.metrics import AGENT_ROUTES, SECURITY_EVENTS
from packages.observability.tracing import current_trace, span
from services.agent.router import RouteDecision, route
from services.agent.tools import get_registry
from services.rag.injection import scan_user_message
from services.rag.service import get_rag_service

MAX_CONTENT_ATTEMPTS = 2


class AgentState(TypedDict, total=False):
    message: str
    principal: Principal
    last_sku: str | None
    decision: RouteDecision
    route: str
    answer: str
    citations: list[dict[str, Any]]
    products: list[dict[str, Any]]
    notices: list[str]
    pending_approval: dict[str, Any] | None
    draft: dict[str, Any]
    validation: dict[str, Any]
    guidelines: list[str]
    attempts: int
    product: dict[str, Any]


def _t(lang: str, en: str, sk: str) -> str:
    return sk if lang in ("sk", "cs") else en


def _exec(state: AgentState, tool: str, args: dict[str, Any]) -> tuple[Any, str | None]:
    result, rec = get_registry().execute(tool, args, state["principal"])
    if rec.outcome == "success":
        return result, None
    return None, rec.error or rec.outcome


# ------------------------------------------------------------------------------------ nodes
def guard(state: AgentState) -> AgentState:
    scan = scan_user_message(state["message"])
    notices = list(state.get("notices", []))
    if scan.flagged:
        SECURITY_EVENTS.labels(event="user_jailbreak_attempt").inc()
        notices.append(
            "Your message contained instruction-override phrasing. Security rules and your role's "
            "permissions still apply."
        )
    return {"notices": notices}


def route_node(state: AgentState) -> AgentState:
    with span("route", kind="internal") as a:
        decision = route(state["message"], state["principal"], state.get("last_sku"))
        a.update(route=decision.route, reason=decision.reason, router=decision.router)
    AGENT_ROUTES.labels(route=decision.route).inc()
    trace = current_trace()
    if trace is not None:
        trace.route = decision.route
    return {"decision": decision, "route": decision.route}


def search_node(state: AgentState) -> AgentState:
    d = state["decision"]
    res, err = _exec(state, "search_products", {"query": state["message"], "limit": 8})
    if err:
        return {"answer": _t(d.language, f"Product search failed: {err}", f"Vyhľadávanie zlyhalo: {err}")}
    hits = res["results"]
    filters = res["query"]["filters"]
    active = {k: v for k, v in filters.items() if v not in (None, "") and k != "language"}
    if not hits:
        return {
            "answer": _t(
                d.language,
                "No products match these criteria. Try relaxing the price or colour.",
                "Kritériám nevyhovujú žiadne produkty. Skúste zmierniť cenu alebo farbu.",
            ),
            "products": [],
        }
    lines = []
    for h in hits[:8]:
        stock = f", {h['stock_available']} pcs available" if h["stock_available"] is not None else ""
        lines.append(f"- **{h['sku']}** {h['name']} – {h['price_eur']:.2f} EUR{stock}")
    head = _t(
        d.language, f"I found {len(hits)} matching products", f"Našiel som {len(hits)} vhodných produktov"
    )
    if active:
        head += " (filters: " + ", ".join(f"{k}={v}" for k, v in active.items()) + ")"
    if res.get("relaxed_filters"):
        head += f". Relaxed: {', '.join(res['relaxed_filters'])}"
    stages = res["stages"]
    method = []
    if "exact_match" in stages:
        method.append(f"exact code lookup ({stages['exact_match']['code']})")
    else:
        if "bm25" in stages:
            method.append(f"BM25 {stages['bm25']['hits']} hits")
        if "vector" in stages:
            method.append(f"vector {stages['vector']['hits']} hits")
        if "fusion" in stages:
            method.append(stages["fusion"]["method"])
        if "rerank" in stages:
            method.append(f"reranked by {stages['rerank']['model']}")
    return {
        "answer": f"{head}:\n" + "\n".join(lines) + f"\n\n_Retrieval: {' → '.join(method)}._",
        "products": hits,
    }


def rag_node(state: AgentState) -> AgentState:
    principal = state["principal"]
    from packages.core.security import Permission

    if not principal.can(Permission.DOCS_READ):
        return {"answer": "You are not allowed to search internal documents."}
    rag = get_rag_service()
    with span("rag_answer", kind="internal"):
        result = rag.answer(state["message"], principal)
    trace = current_trace()
    if trace is not None:
        trace.tool_calls.append(
            {
                "tool": "search_company_documents",
                "args": {"query": state["message"]},
                "outcome": "success",
                "side_effect": "read",
                "duration_ms": 0,
                "result_preview": f"{len(result.chunks)} chunks, {result.filtered_by_rbac} filtered by RBAC",
                "error": "",
            }
        )
    notices = list(state.get("notices", []))
    if result.filtered_by_rbac:
        notices.append(
            f"{result.filtered_by_rbac} relevant document chunk(s) were excluded because your role "
            f"('{principal.role.value}') is not permitted to read them."
        )
    if result.injection_flags:
        notices.append(
            f"{result.injection_flags} retrieved chunk(s) contained prompt-injection patterns; the "
            "instructions were removed and treated as untrusted data."
        )
    return {
        "answer": result.answer,
        "citations": [c.model_dump() | {"label": c.label()} for c in result.citations],
        "notices": notices,
    }


def stock_node(state: AgentState) -> AgentState:
    d = state["decision"]
    res, err = _exec(state, "get_product_stock", {"sku": d.sku})
    if err:
        return {
            "answer": _t(
                d.language,
                f"I cannot provide stock for {d.sku}: {err}",
                f"Sklad pre {d.sku} nemôžem poskytnúť: {err}",
            )
        }
    wh = ", ".join(f"{k}: {v}" for k, v in res["by_warehouse"].items())
    return {
        "answer": _t(
            d.language,
            f"**{res['sku']}** has **{res['total_available']} pieces** available ({wh}). "
            f"Supplier lead time for more: {res['lead_time_days']} days. Source: ERP stock table "
            f"(as of {res['as_of'][:16].replace('T', ' ')} UTC).",
            f"**{res['sku']}** má k dispozícii **{res['total_available']} ks** ({wh}). "
            f"Dodacia lehota dodávateľa: {res['lead_time_days']} dní. Zdroj: skladová tabuľka ERP.",
        ),
        "last_sku": res["sku"],
    }


def price_node(state: AgentState) -> AgentState:
    d = state["decision"]
    res, err = _exec(state, "get_product_price", {"sku": d.sku, "quantity": d.args.get("quantity", 1)})
    if err:
        return {"answer": f"I cannot provide pricing for {d.sku}: {err}"}
    tiers = ", ".join(f"{t['min_qty']}+ pcs: {t['unit_price_eur']:.2f}" for t in res["tiers"])
    return {
        "answer": f"**{res['sku']}** list price {res['list_price_eur']:.2f} EUR (excl. VAT). Volume tiers: "
        f"{tiers}. For {res['quantity']} pcs: {res['unit_price_eur']:.2f} EUR/pc, total "
        f"{res['total_eur']:.2f} EUR.",
        "last_sku": res["sku"],
    }


def product_info_node(state: AgentState) -> AgentState:
    d = state["decision"]
    res, err = _exec(state, "get_product", {"sku": d.sku})
    if err:
        return {"answer": f"{err}"}
    stock = f" Available: {res['stock_available']} pcs." if res["stock_available"] is not None else ""
    cap = f", {res['capacity_ml']} ml" if res["capacity_ml"] else ""
    name = res["name_sk"] if d.language in ("sk", "cs") else res["name_en"]
    return {
        "answer": f"**{res['sku']}** – {name} ({res['material']}{cap}, {res['color']}), "
        f"{res['price_eur']:.2f} EUR, supplier {res['supplier']} ({res['supplier_sku']}).{stock}",
        "products": [res],
        "last_sku": res["sku"],
    }


def supplier_node(state: AgentState) -> AgentState:
    d = state["decision"]
    res, err = _exec(state, "get_supplier_info", {"sku": d.sku})
    if err:
        return {"answer": f"I cannot provide supplier information: {err}"}
    express = (
        "supports urgent/express delivery"
        if res["urgent_delivery_supported"]
        else "does NOT offer express delivery"
    )
    return {
        "answer": f"{d.sku} is supplied by **{res['name']}** ({res['code']}, {res['country']}). Standard lead "
        f"time {res['default_lead_time_days']} days; {express}. {res['notes']}"
    }


ANALYTICS_TEMPLATES = [
    (
        re.compile(
            r"\b(top|best[- ]selling|most sold)\b.*\b(products?|items?)\b|\brevenue by product\b", re.I
        ),
        "SELECT p.sku, p.name_en, SUM(oi.quantity) AS units, ROUND(CAST(SUM(oi.quantity * oi.unit_price_eur) AS NUMERIC), 2) AS revenue_eur "
        "FROM order_items oi JOIN products p ON p.id = oi.product_id JOIN orders o ON o.id = oi.order_id "
        "WHERE o.status <> 'cancelled' GROUP BY p.sku, p.name_en ORDER BY revenue_eur DESC LIMIT {n}",
    ),
    (
        re.compile(r"\b(revenue|sales|turnover|trzby)\b.*\b(category|categories|kategori\w*)\b", re.I),
        "SELECT p.category, ROUND(CAST(SUM(oi.quantity * oi.unit_price_eur) AS NUMERIC), 2) AS revenue_eur, COUNT(DISTINCT o.id) AS orders "
        "FROM order_items oi JOIN products p ON p.id = oi.product_id JOIN orders o ON o.id = oi.order_id "
        "WHERE o.status <> 'cancelled' GROUP BY p.category ORDER BY revenue_eur DESC",
    ),
    (
        re.compile(r"\b(revenue|sales)\b.*\bsuppliers?\b", re.I),
        "SELECT s.code, s.name, ROUND(CAST(SUM(oi.quantity * oi.unit_price_eur) AS NUMERIC), 2) AS revenue_eur FROM order_items oi "
        "JOIN products p ON p.id = oi.product_id JOIN suppliers s ON s.id = p.supplier_id GROUP BY s.code, s.name "
        "ORDER BY revenue_eur DESC",
    ),
]


def analytics_node(state: AgentState) -> AgentState:
    msg = state["message"]
    sql = None
    for pattern, template in ANALYTICS_TEMPLATES:
        if pattern.search(msg):
            m = re.search(r"\btop (\d+)\b", msg, re.I)
            sql = template.format(n=int(m.group(1)) if m else 10)
            break
    if sql is None:
        return {
            "answer": "In mock-LLM mode analytics supports: top products by revenue, revenue by category, revenue "
            "by supplier. With a real LLM configured, free-form questions are translated to SQL and then "
            "pass the same AST validation, allowlist, row limit and read-only role."
        }
    res, err = _exec(state, "run_analytics_query", {"sql": sql})
    if err:
        return {"answer": f"Analytics query not executed: {err}"}
    header = " | ".join(res["columns"])
    rows = "\n".join(" | ".join(str(v) for v in r) for r in res["rows"][:10])
    return {
        "answer": f"```\n{header}\n{rows}\n```\n_Validated read-only SQL ({res['row_count']} rows):_ `{res['sql']}`"
    }


def translate_node(state: AgentState) -> AgentState:
    d = state["decision"]
    target = d.args.get("target", "cs")
    res, err = _exec(state, "translate_product_description", {"sku": d.sku, "source": "sk", "target": target})
    if err:
        return {"answer": f"I cannot translate this product text: {err}"}
    val, _ = _exec(state, "validate_product_data", {"sku": d.sku, "language": target, "text": res["text"]})
    prop, perr = _exec(
        state,
        "prepare_product_update",
        {
            "sku": d.sku,
            "field": f"description_{target}",
            "new_value": res["text"],
            "rationale": f"Translation sk→{target} requested in chat",
            "validation": val or {},
        },
    )
    tail = (
        f"\n\nProposal #{prop['request_id']} is **pending approval**." if prop else f"\n\n(Not saved: {perr})"
    )
    return {
        "answer": f"Draft {target.upper()} translation for {d.sku}:\n\n> {res['text']}{tail}",
        "pending_approval": prop,
    }


def content_gather(state: AgentState) -> AgentState:
    d = state["decision"]
    product, err = _exec(state, "get_product", {"sku": d.sku})
    if err:
        return {"answer": f"Cannot generate content: {err}", "attempts": MAX_CONTENT_ATTEMPTS + 1}
    docs, _ = _exec(
        state,
        "search_company_documents",
        {"query": "brand guidelines tone of voice product copy guidelines description structure length"},
    )
    guidelines: list[str] = []
    for chunk in (docs or {}).get("chunks", []):
        if chunk["source"] in ("brand_guidelines.docx", "product_copy_guidelines.docx"):
            guidelines.extend(s.strip() for s in re.split(r"(?<=[.])\s+", chunk["text"]) if 20 < len(s) < 220)
    citations = [
        {
            "ref": i + 1,
            "source": c["source"],
            "doc_type": c["doc_type"],
            "locator": c["locator"],
            "snippet": c["text"][:200],
            "score": c["score"],
            "label": f"{c['source']} | Section: {c['locator'].get('section', '')}",
        }
        for i, c in enumerate(
            c
            for c in (docs or {}).get("chunks", [])
            if c["source"] in ("brand_guidelines.docx", "product_copy_guidelines.docx")
        )
    ][:3]
    return {"product": product, "guidelines": guidelines[:12], "attempts": 0, "citations": citations}


def content_generate(state: AgentState) -> AgentState:
    d = state["decision"]
    lang = d.args.get("language", "sk")
    draft, err = _exec(
        state,
        "generate_product_description",
        {"sku": d.sku, "language": lang, "guidelines": state.get("guidelines", [])},
    )
    if err:
        return {
            "answer": f"Cannot generate content: {err}",
            "attempts": MAX_CONTENT_ATTEMPTS + 1,
            "draft": {},
        }
    return {"draft": draft, "attempts": state.get("attempts", 0) + 1}


def content_validate(state: AgentState) -> AgentState:
    d = state["decision"]
    if not state.get("draft"):
        return {}
    val, err = _exec(
        state,
        "validate_product_data",
        {"sku": d.sku, "language": state["draft"]["language"], "text": state["draft"]["text"]},
    )
    return {"validation": val or {"valid": False, "errors": [err]}}


def content_after_validate(state: AgentState) -> str:
    if not state.get("draft") or state.get("attempts", 0) > MAX_CONTENT_ATTEMPTS:
        return "finalize"
    if state.get("validation", {}).get("valid"):
        return "content_propose"
    return "content_generate" if state.get("attempts", 0) < MAX_CONTENT_ATTEMPTS else "finalize"


def content_propose(state: AgentState) -> AgentState:
    state["decision"]
    draft, val = state["draft"], state["validation"]
    prop, err = _exec(
        state,
        "prepare_product_update",
        {
            "sku": state["product"]["sku"],
            "field": f"description_{draft['language']}",
            "new_value": draft["text"],
            "rationale": f"Generated in chat: {state['message'][:200]}",
            "validation": val,
        },
    )
    checks = ", ".join(k for k, v in val.get("checks", {}).items() if v)
    warn = f"\nWarnings: {'; '.join(val['warnings'])}" if val.get("warnings") else ""
    if err:
        return {"answer": f"Draft generated but not submitted: {err}\n\n> {draft['text']}"}
    return {
        "answer": f"Draft {draft['language'].upper()} description for **{state['product']['sku']}** "
        f"({len(draft['text'])} chars, validation passed: {checks}).{warn}\n\n> {draft['text']}\n\n"
        f"🔒 Nothing was written to the product database. Change request **#{prop['request_id']}** "
        f"is **pending approval** by an administrator (payload hash `{prop['payload_hash'][:12]}…`, "
        f"base version {prop['base_version']}).",
        "pending_approval": prop,
        "last_sku": state["product"]["sku"],
    }


def refuse_node(state: AgentState) -> AgentState:
    SECURITY_EVENTS.labels(event="sensitive_request_refused").inc()
    return {
        "answer": "I can't help with exporting customer records, personal data, credentials or system "
        "internals. The assistant has no tool that can access customer personal data; bulk exports "
        "require written approval of the Data Protection Officer (Company Policy, section 8)."
    }


def general_node(state: AgentState) -> AgentState:
    from services.llm.factory import get_llm
    from services.llm.prompts import SYSTEM_BASE

    d = state["decision"]
    res = get_llm().complete(
        [{"role": "system", "content": SYSTEM_BASE}, {"role": "user", "content": state["message"]}],
        task="general",
        payload={"language": d.language},
    )
    return {"answer": res.text}


def finalize(state: AgentState) -> AgentState:
    if state.get("answer"):
        return {}
    val = state.get("validation", {})
    if state.get("draft") and not val.get("valid", True):
        return {
            "answer": "The generated draft failed validation after "
            f"{state.get('attempts')} attempts and was NOT submitted: {'; '.join(val.get('errors', []))}"
            f"\n\n> {state['draft']['text']}"
        }
    return {"answer": "No answer could be produced."}


ROUTE_NODES = {
    "search": search_node,
    "rag": rag_node,
    "stock": stock_node,
    "price": price_node,
    "product_info": product_info_node,
    "supplier": supplier_node,
    "analytics": analytics_node,
    "translate": translate_node,
    "refuse": refuse_node,
    "general": general_node,
}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("guard", guard)
    g.add_node("route", route_node)
    for name, fn in ROUTE_NODES.items():
        g.add_node(name, fn)
        g.add_edge(name, "finalize")
    g.add_node("content_gather", content_gather)
    g.add_node("content_generate", content_generate)
    g.add_node("content_validate", content_validate)
    g.add_node("content_propose", content_propose)
    g.add_node("finalize", finalize)
    g.set_entry_point("guard")
    g.add_edge("guard", "route")
    g.add_conditional_edges(
        "route",
        lambda s: "content_gather" if s["route"] == "content" else s["route"],
        {**{k: k for k in ROUTE_NODES}, "content_gather": "content_gather"},
    )
    g.add_conditional_edges(
        "content_gather",
        lambda s: "finalize" if s.get("answer") else "content_generate",
        {"finalize": "finalize", "content_generate": "content_generate"},
    )
    g.add_edge("content_generate", "content_validate")
    g.add_conditional_edges(
        "content_validate",
        content_after_validate,
        {
            "content_propose": "content_propose",
            "content_generate": "content_generate",
            "finalize": "finalize",
        },
    )
    g.add_edge("content_propose", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
