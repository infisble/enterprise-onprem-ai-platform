"""Chat entry point: runs the agent graph inside a trace and persists the trace."""

from __future__ import annotations

from typing import Any

from packages.core.cache import get_store
from packages.core.config import get_settings
from packages.core.db import session_scope
from packages.core.models import RequestTrace
from packages.core.security import Principal
from packages.observability.logging import get_logger
from packages.observability.metrics import ERRORS, WINDOW
from packages.observability.tracing import Trace, start_trace
from packages.schemas.chat import ChatResponse, Citation, ToolCallRecord
from services.agent.graph import get_graph

log = get_logger(__name__)
SESSION_TTL_S = 1800


def persist_trace(trace: Trace, kind: str = "chat") -> None:
    summary = trace.summary()
    WINDOW.add(
        total_ms=summary["total_ms"],
        llm_ms=summary["llm_ms"],
        retrieval_ms=summary["retrieval_ms"],
        output_tokens=trace.output_tokens,
        route=trace.route or kind,
        error=bool(trace.error),
    )
    log.info("request completed", extra={"trace": summary, "kind": kind})
    try:
        with session_scope() as s:
            s.add(
                RequestTrace(
                    request_id=trace.request_id,
                    username=trace.username,
                    role=trace.role,
                    kind=kind,
                    query=trace.query[:2000],
                    route=trace.route,
                    llm_provider=trace.llm_provider,
                    model=trace.model,
                    total_ms=summary["total_ms"],
                    input_tokens=trace.input_tokens,
                    output_tokens=trace.output_tokens,
                    spans=trace.spans,
                    tool_calls=trace.tool_calls,
                    error=trace.error,
                )
            )
    except Exception:  # tracing must never break the request
        log.exception("failed to persist trace")


def run_chat(message: str, principal: Principal, conversation_id: str | None = None) -> ChatResponse:
    store = get_store()
    session_key = f"session:{principal.username}:{conversation_id or 'default'}"
    session = store.get_json(session_key) or {}
    with start_trace(
        username=principal.username, role=principal.role.value, kind="chat", query=message
    ) as trace:
        try:
            state: dict[str, Any] = get_graph().invoke(
                {
                    "message": message,
                    "principal": principal,
                    "last_sku": session.get("last_sku"),
                    "notices": [],
                }
            )
        except Exception as exc:
            trace.error = f"{type(exc).__name__}: {exc}"
            ERRORS.labels(component="agent").inc()
            persist_trace(trace)
            raise
        if state.get("last_sku"):
            store.set_json(session_key, {"last_sku": state["last_sku"]}, ttl_s=SESSION_TTL_S)
        tool_calls = [ToolCallRecord(**t) for t in trace.tool_calls]
        notices = list(state.get("notices", []))
        notices += [f"Tool '{t.tool}' denied: {t.error}" for t in tool_calls if t.outcome == "denied"]
        s = get_settings()
        response = ChatResponse(
            request_id=trace.request_id,
            answer=state.get("answer", ""),
            route=state.get("route", ""),
            citations=[
                Citation(**{k: v for k, v in c.items() if k != "label"}) for c in state.get("citations", [])
            ],
            tool_calls=tool_calls,
            products=state.get("products", [])[:8],
            pending_approval=state.get("pending_approval"),
            security_notices=notices,
            latency={k: v for k, v in trace.summary().items() if k.endswith("_ms")},
            llm={
                "provider": trace.llm_provider or s.llm_provider,
                "model": trace.model or s.active_model,
                "input_tokens": trace.input_tokens,
                "output_tokens": trace.output_tokens,
                "router": state["decision"].router if state.get("decision") else "rules",
                "route_reason": state["decision"].reason if state.get("decision") else "",
            },
        )
        persist_trace(trace)
        return response
