"""Lightweight request tracing.

Every chat/search request gets a `Trace` holding timed spans (retrieval, rerank, sql, llm, tool),
token usage and tool calls. Traces are logged as structured JSON, exported to Prometheus
histograms and persisted to SQL so the UI can show "Agent Activity". The span model maps 1:1
onto OpenTelemetry if an OTLP collector is added later.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from packages.observability.metrics import ERRORS, STAGE_LATENCY


@dataclass
class Trace:
    username: str = "anonymous"
    role: str = "anonymous"
    kind: str = "chat"
    query: str = ""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    route: str = ""
    llm_provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float | None = None
    spans: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    _t0: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 2)

    def stage_ms(self, kind: str) -> float:
        return round(sum(s["duration_ms"] for s in self.spans if s["kind"] == kind), 2)

    def summary(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.username,
            "role": self.role,
            "route": self.route,
            "llm_provider": self.llm_provider,
            "model": self.model,
            "retrieval_ms": self.stage_ms("retrieval"),
            "rerank_ms": self.stage_ms("rerank"),
            "sql_ms": self.stage_ms("sql"),
            "llm_ms": self.stage_ms("llm"),
            "tool_ms": self.stage_ms("tool"),
            "ttft_ms": self.ttft_ms,
            "total_ms": self.elapsed_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": [t["tool"] for t in self.tool_calls],
            "error": self.error,
        }


_current: ContextVar[Trace | None] = ContextVar("current_trace", default=None)


def current_trace() -> Trace | None:
    return _current.get()


@contextmanager
def start_trace(**kwargs: Any) -> Iterator[Trace]:
    # Reuse the HTTP request id so logs, traces and audit records share one correlation id.
    from packages.observability.logging import request_ctx

    if "request_id" not in kwargs and (rid := request_ctx.get().get("request_id")):
        kwargs["request_id"] = rid
    trace = Trace(**kwargs)
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)


@contextmanager
def span(name: str, kind: str = "internal", **attrs: Any) -> Iterator[dict[str, Any]]:
    """Time a pipeline stage. `kind` is one of retrieval|rerank|sql|llm|tool|internal."""
    record: dict[str, Any] = {"name": name, "kind": kind, "attrs": dict(attrs)}
    trace = _current.get()
    if trace is not None:
        record["start_ms"] = trace.elapsed_ms
    t0 = time.perf_counter()
    try:
        yield record["attrs"]
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        ERRORS.labels(component=kind).inc()
        raise
    finally:
        record["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if kind != "internal":
            STAGE_LATENCY.labels(stage=kind if kind != "tool" else f"tool:{name}").observe(
                record["duration_ms"] / 1000
            )
        if trace is not None:
            trace.spans.append(record)
