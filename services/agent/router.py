"""Intent routing.

Default: deterministic rules (fast, free, testable, cannot be talked into a route by a document).
Optional: LLM tool-calling router (`LLM_ROUTING=true` with a real provider). The LLM is only offered
the tools the caller's role is allowed to use, and its choice is still executed through the governed
tool executor - so a wrong or manipulated routing decision can never exceed the user's privileges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from packages.core.config import get_settings
from packages.core.security import Principal
from packages.core.text import CONCEPT_LOOKUP, PRODUCT_TYPES, detect_language, normalize, tokenize
from packages.observability.logging import get_logger

log = get_logger(__name__)

SKU_RE = re.compile(r"\b(NP-\d{5}|\d{5}\.\d{2}|PT-\d{4}-[A-Z]{2}|ECO-\d{4}-\d)\b", re.I)

SENSITIVE = re.compile(
    r"\b(export|dump|download|list|show|give me|send)\b.{0,40}\b(all )?(customers?|clients?|customer (records|data|"
    r"emails?|list)|e-?mail addresses|passwords?|credentials)\b|\bsystem prompt\b|\bzoznam zakaznikov\b|"
    r"\bexportuj\b.{0,30}\bzakaznik",
    re.I,
)
ANALYTICS = re.compile(
    r"\b(top \d+|best[- ]selling|revenue|turnover|trzby|trzieb|obrat|sales by|orders per|"
    r"most sold|analytics|sql)\b",
    re.I,
)
CONTENT_VERB = re.compile(
    r"\b(generate|write|create|draft|rewrite|improve|update|vygeneruj|napis|vytvor|"
    r"aktualizuj|priprav|navrhni)\b",
    re.I,
)
CONTENT_NOUN = re.compile(r"\b(description|popis|copy|text|product text)\b", re.I)
TRANSLATE = re.compile(r"\b(translate|translation|preloz|preklad|preloz|prelozit|prelozte)\b", re.I)
STOCK = re.compile(
    r"\b(how many|stock|in stock|available|availability|pieces|pcs|kolko|kolik|sklad|skladom|"
    r"skladem|kusov|kusu|dostupn)\w*",
    re.I,
)
PRICE = re.compile(
    r"\b(price|prices|cost|costs|how much|quote|cena|ceny|kolko stoji|kolik stoji|pricing)\b", re.I
)
SUPPLIER = re.compile(r"\b(supplier|vendor|dodavatel\w*|who supplies|eurogift|promotex|ecoline)\b", re.I)
RAG_WORDS = re.compile(
    r"\b(policy|policies|rule|rules|procedure|process|guideline|guidelines|terms|handbook|faq|returns?|complaints?|"
    r"shipping|delivery times?|express|surcharges?|urgent|notice|summari[sz]e|what is our|how do we|how should|"
    r"allowed|brand colou?rs?|tone of voice|smernic\w*|pravidl\w*|postup\w*|reklamaci\w*|vratenie|doprav\w*|"
    r"minimalna objednavka|minimum order|warehouses?|approves?|margin|discount)\b",
    re.I,
)
SEARCH_VERB = re.compile(
    r"\b(find|search|looking for|need|recommend|suggest|show me|gift|gifts|ideas?|hladam|"
    r"potrebujem|odporuc|darcek\w*|hledam|potrebuji)\b",
    re.I,
)
GREETING = re.compile(r"^\s*(hi|hello|hey|ahoj|dobry den|cau|help|what can you do)\b", re.I)


@dataclass
class RouteDecision:
    route: str
    reason: str
    sku: str | None = None
    language: str = "en"
    args: dict[str, Any] = field(default_factory=dict)
    router: str = "rules"


def _content_language(norm: str, default: str) -> str:
    if re.search(r"\b(slovak|slovensk\w*|sk)\b", norm):
        return "sk"
    if re.search(r"\b(czech|cesk\w*|cs|cz)\b", norm):
        return "cs"
    if re.search(r"\b(english|anglick\w*|en)\b", norm):
        return "en"
    return default


def route_rules(message: str, last_sku: str | None = None) -> RouteDecision:
    norm = normalize(message)
    lang = detect_language(message)
    m = SKU_RE.search(message)
    sku = m.group(1).upper() if m and not m.group(1)[0].isdigit() else (m.group(1) if m else None)
    tokens = tokenize(message)
    has_product_noun = any(t in PRODUCT_TYPES for t in tokens)
    has_concept = any(t in CONCEPT_LOOKUP for t in tokens)
    # Follow-ups ("how many of it are in stock?") reuse the SKU from session state, but only when the
    # message refers back to a product and is not itself a new search ("bottles that are in stock").
    refers_back = re.search(
        r"\b(it|this one|that one|this product|that product|the same|ten|tento|tomto|toho)\b", norm
    )
    is_new_search = has_product_noun or bool(SEARCH_VERB.search(norm))
    ref_sku = sku or (last_sku if refers_back and not is_new_search else None)

    if SENSITIVE.search(norm):
        return RouteDecision("refuse", "request for restricted personal data / system internals", sku, lang)
    if CONTENT_VERB.search(norm) and CONTENT_NOUN.search(norm) and ref_sku:
        return RouteDecision(
            "content",
            "content generation for a product",
            ref_sku,
            lang,
            {"language": _content_language(norm, "sk")},
        )
    if TRANSLATE.search(norm) and ref_sku:
        return RouteDecision(
            "translate",
            "translation of a product text",
            ref_sku,
            lang,
            {"target": _content_language(norm.replace("from slovak", ""), "cs")},
        )
    if ANALYTICS.search(norm) and not ref_sku:
        return RouteDecision("analytics", "aggregate business analytics question", None, lang)
    if ref_sku and PRICE.search(norm):
        m_qty = re.search(r"\b(\d{2,6})\s*(pcs|pieces|ks|kusov|units)\b", norm)
        return RouteDecision(
            "price",
            "price question with a product code",
            ref_sku,
            lang,
            {"quantity": int(m_qty.group(1)) if m_qty else 1},
        )
    if ref_sku and STOCK.search(norm):
        return RouteDecision("stock", "stock question with a product code", ref_sku, lang)
    if SUPPLIER.search(norm) and ref_sku and not RAG_WORDS.search(norm):
        return RouteDecision("supplier", "supplier question for a product", ref_sku, lang)
    if sku:
        if re.search(r"\b(find|search|product)\b", norm) and not re.search(r"\b(detail|info|about)\b", norm):
            return RouteDecision("search", "product code lookup", sku, lang)
        return RouteDecision("product_info", "product code mentioned", sku, lang)
    if RAG_WORDS.search(norm) and not (has_product_noun and SEARCH_VERB.search(norm)):
        return RouteDecision("rag", "policy / procedure / document question", None, lang)
    if SUPPLIER.search(norm):
        return RouteDecision("rag", "supplier question without product code -> documents", None, lang)
    if (
        has_product_noun
        or (has_concept and SEARCH_VERB.search(norm))
        or SEARCH_VERB.search(norm)
        or re.search(r"\b(under|below|do|pod)\s+\d", norm)
    ):
        return RouteDecision("search", "product discovery request", None, lang)
    if GREETING.search(norm):
        return RouteDecision("general", "greeting / capabilities", None, lang)
    if message.strip().endswith("?") or re.match(
        r"^(what|how|why|when|who|where|which|co|ako|preco|kedy|kto|kde|jak|proc)\b", norm
    ):
        return RouteDecision("rag", "open question -> knowledge base", None, lang)
    return RouteDecision("general", "no specific intent", None, lang)


TOOL_TO_ROUTE = {
    "search_products": "search",
    "get_product": "product_info",
    "get_product_stock": "stock",
    "get_product_price": "price",
    "search_company_documents": "rag",
    "get_supplier_info": "supplier",
    "generate_product_description": "content",
    "translate_product_description": "translate",
    "run_analytics_query": "analytics",
}


def route(message: str, principal: Principal, last_sku: str | None = None) -> RouteDecision:
    rules = route_rules(message, last_sku)
    s = get_settings()
    if not s.llm_routing or s.llm_provider == "mock" or rules.route == "refuse":
        return rules
    from services.agent.tools import get_registry
    from services.llm.factory import get_llm
    from services.llm.prompts import router_messages

    tools = [t.openai_schema() for t in get_registry().allowed_for(principal) if t.name in TOOL_TO_ROUTE]
    try:
        result = get_llm().complete(router_messages(message), task="route", tools=tools, max_tokens=200)
        if result.tool_calls:
            call = result.tool_calls[0]
            args = call.get("arguments", {})
            return RouteDecision(
                TOOL_TO_ROUTE.get(call["name"], rules.route),
                f"llm tool choice: {call['name']}",
                (args.get("sku") or rules.sku),
                rules.language,
                {**rules.args, **args},
                router="llm",
            )
    except Exception as exc:  # fall back to rules on any provider problem
        log.warning("LLM routing failed, using rules", extra={"error": str(exc)})
    return rules
