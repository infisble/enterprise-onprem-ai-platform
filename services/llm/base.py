"""Provider-agnostic LLM interface.

Application code never imports an SDK directly; it calls `LLMProvider.complete()` with an
OpenAI-style message list. This is what allows the same code to run against a deterministic mock
(laptop / CI), OpenAI (prototyping) or a vLLM server on NovaPromo's own NVIDIA GPU (production).

`payload` carries the structured data the prompt was rendered from. Real providers ignore it;
the mock provider uses it to produce grounded, deterministic output without a model.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from packages.observability.metrics import LLM_LATENCY, LLM_TOKENS, LLM_TTFT
from packages.observability.tracing import current_trace, span


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"


def estimate_tokens(text: str) -> int:
    # ~4 chars/token for English, a bit less for Slovak/Czech; good enough for accounting.
    return max(1, len(text) // 4)


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        task: str,
        payload: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        with span(f"llm:{task}", kind="llm", provider=self.name, model=self.model) as attrs:
            t0 = time.perf_counter()
            result = self._complete(
                messages,
                task=task,
                payload=payload or {},
                tools=tools,
                json_schema=json_schema,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            attrs.update(
                input_tokens=result.input_tokens, output_tokens=result.output_tokens, ttft_ms=result.ttft_ms
            )
        LLM_LATENCY.labels(self.name, self.model, task).observe(result.latency_ms / 1000)
        LLM_TOKENS.labels(self.name, self.model, "input").inc(result.input_tokens)
        LLM_TOKENS.labels(self.name, self.model, "output").inc(result.output_tokens)
        if result.ttft_ms is not None:
            LLM_TTFT.labels(self.name, self.model).observe(result.ttft_ms / 1000)
        trace = current_trace()
        if trace is not None:
            trace.llm_provider, trace.model = self.name, self.model
            trace.input_tokens += result.input_tokens
            trace.output_tokens += result.output_tokens
            if trace.ttft_ms is None and result.ttft_ms is not None:
                trace.ttft_ms = result.ttft_ms
        return result

    @abstractmethod
    def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        task: str,
        payload: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        json_schema: dict[str, Any] | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> LLMResult: ...

    def health(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "ok": True}

    @property
    def supports_tools(self) -> bool:
        return False
