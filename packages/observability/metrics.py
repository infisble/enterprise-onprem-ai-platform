"""Prometheus metrics + a small rolling window used by the UI's System Metrics page."""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque

from prometheus_client import Counter, Gauge, Histogram

_LAT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP latency", ["method", "route"], buckets=_LAT_BUCKETS
)
LLM_LATENCY = Histogram(
    "llm_request_duration_seconds", "LLM call latency", ["provider", "model", "task"], buckets=_LAT_BUCKETS
)
LLM_TTFT = Histogram(
    "llm_time_to_first_token_seconds", "Time to first token", ["provider", "model"], buckets=_LAT_BUCKETS
)
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["provider", "model", "direction"])
STAGE_LATENCY = Histogram(
    "pipeline_stage_duration_seconds",
    "Latency of retrieval / rerank / sql / tool stages",
    ["stage"],
    buckets=_LAT_BUCKETS,
)
TOOL_CALLS = Counter("agent_tool_calls_total", "Agent tool calls", ["tool", "outcome"])
AGENT_ROUTES = Counter("agent_routes_total", "Agent routing decisions", ["route"])
ERRORS = Counter("errors_total", "Errors by component", ["component"])
SECURITY_EVENTS = Counter("security_events_total", "Denied tools, filtered chunks, injections", ["event"])
BATCH_ITEMS = Counter("batch_items_total", "Batch items processed", ["job_type", "status"])
BATCH_QUEUE = Gauge("batch_queue_size", "Items waiting in the batch queue")
BATCH_RUNNING = Gauge("batch_jobs_running", "Batch jobs currently running")
# GPU utilisation/memory come from the NVIDIA DCGM exporter; the app only mirrors vLLM's KV-cache usage.
KV_CACHE = Gauge("llm_kv_cache_usage_ratio", "vLLM KV-cache usage (0-1), mirrored from vLLM /metrics")


class RollingWindow:
    """Last N minutes of request samples for human-readable dashboards (not a TSDB)."""

    def __init__(self, horizon_s: int = 900) -> None:
        self.horizon_s = horizon_s
        self._samples: deque[dict] = deque()
        self._lock = threading.Lock()

    def add(self, **sample: float | str | bool) -> None:
        sample["t"] = time.time()
        with self._lock:
            self._samples.append(sample)
            self._trim()

    def _trim(self) -> None:
        cutoff = time.time() - self.horizon_s
        while self._samples and self._samples[0]["t"] < cutoff:
            self._samples.popleft()

    def summary(self) -> dict:
        with self._lock:
            self._trim()
            samples = list(self._samples)
        if not samples:
            return {"requests": 0, "requests_per_min": 0.0}

        def pct(values: list[float], q: float) -> float:
            if not values:
                return 0.0
            values = sorted(values)
            return round(values[min(len(values) - 1, int(q * len(values)))], 1)

        total = [float(s["total_ms"]) for s in samples]
        llm = [float(s["llm_ms"]) for s in samples if s.get("llm_ms")]
        retr = [float(s["retrieval_ms"]) for s in samples if s.get("retrieval_ms")]
        out_tokens = sum(int(s.get("output_tokens", 0)) for s in samples)
        llm_seconds = sum(llm) / 1000 or 1.0
        span_min = max(1.0, (time.time() - samples[0]["t"]) / 60)
        routes: dict[str, int] = {}
        for s in samples:
            routes[str(s.get("route", ""))] = routes.get(str(s.get("route", "")), 0) + 1
        return {
            "requests": len(samples),
            "requests_per_min": round(len(samples) / span_min, 2),
            "latency_ms": {
                "p50": pct(total, 0.5),
                "p95": pct(total, 0.95),
                "mean": round(statistics.mean(total), 1),
            },
            "llm_latency_ms": {"p50": pct(llm, 0.5), "p95": pct(llm, 0.95)},
            "retrieval_latency_ms": {"p50": pct(retr, 0.5), "p95": pct(retr, 0.95)},
            "output_tokens": out_tokens,
            "tokens_per_sec": round(out_tokens / llm_seconds, 1),
            "error_rate": round(sum(1 for s in samples if s.get("error")) / len(samples), 3),
            "routes": routes,
        }


WINDOW = RollingWindow()
