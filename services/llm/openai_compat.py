"""OpenAI-compatible chat-completions client used for both OpenAI and a self-hosted vLLM server.

vLLM exposes the same `/v1/chat/completions` API, so switching NovaPromo from a hosted model to
its on-prem NVIDIA server is a configuration change (`LLM_PROVIDER=vllm`, `VLLM_BASE_URL=...`).
Streaming is used so we can measure time-to-first-token (TTFT), the latency users actually feel.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import Any

import httpx

from packages.core.config import get_settings
from services.llm.base import LLMProvider, LLMResult, estimate_tokens


class OpenAICompatibleProvider(LLMProvider):
    name = "openai-compatible"

    def __init__(self, base_url: str, model: str, api_key: str, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    @property
    def supports_tools(self) -> bool:
        return True

    def _body(self, messages, tools, json_schema, max_tokens, temperature) -> dict[str, Any]:
        s = get_settings()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or s.llm_max_tokens,
            "temperature": s.llm_temperature if temperature is None else temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "output"),
                    "schema": json_schema,
                    "strict": False,
                },
            }
        return body

    def _complete(self, messages, *, task, payload, tools, json_schema, max_tokens, temperature) -> LLMResult:
        body = self._body(messages, tools, json_schema, max_tokens, temperature)
        if tools:  # tool calls are simpler to parse non-streamed
            r = self._client.post(f"{self.base_url}/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            calls = [
                {
                    "name": c["function"]["name"],
                    "arguments": json.loads(c["function"].get("arguments") or "{}"),
                }
                for c in msg.get("tool_calls") or []
            ]
            usage_nt = data.get("usage", {})
            return LLMResult(
                text=msg.get("content") or "",
                provider=self.name,
                model=self.model,
                tool_calls=calls,
                input_tokens=usage_nt.get("prompt_tokens", 0),
                output_tokens=usage_nt.get("completion_tokens", 0),
                finish_reason=data["choices"][0].get("finish_reason", "stop"),
            )

        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        t0 = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        usage: dict[str, int] = {}
        finish = "stop"
        with self._client.stream("POST", f"{self.base_url}/chat/completions", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                event = json.loads(chunk)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {}).get("content")
                    if delta:
                        if ttft is None:
                            ttft = round((time.perf_counter() - t0) * 1000, 2)
                        parts.append(delta)
                    finish = choice.get("finish_reason") or finish
        text = "".join(parts)
        return LLMResult(
            text=text,
            provider=self.name,
            model=self.model,
            ttft_ms=ttft,
            finish_reason=finish,
            input_tokens=usage.get("prompt_tokens") or sum(estimate_tokens(m["content"]) for m in messages),
            output_tokens=usage.get("completion_tokens") or estimate_tokens(text),
        )

    def health(self) -> dict[str, Any]:
        try:
            r = self._client.get(f"{self.base_url}/models", timeout=3)
            return {"provider": self.name, "model": self.model, "ok": r.status_code == 200}
        except httpx.HTTPError as exc:
            return {"provider": self.name, "model": self.model, "ok": False, "error": str(exc)}


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM server (`vllm serve <model>`), e.g. Qwen2.5-14B-Instruct-AWQ on an NVIDIA L40S/RTX 6000."""

    name = "vllm"

    def gpu_stats(self) -> dict[str, float]:
        """Scrape vLLM's Prometheus endpoint for KV-cache usage and queue depth."""
        root = self.base_url.rsplit("/v1", 1)[0]
        stats: dict[str, float] = {}
        try:
            text = self._client.get(f"{root}/metrics", timeout=2).text
        except httpx.HTTPError:
            return stats
        # Metric names differ across vLLM versions; normalise to one key per signal.
        names = {
            "vllm:kv_cache_usage_perc": "kv_cache_usage",
            "vllm:gpu_cache_usage_perc": "kv_cache_usage",
            "vllm:num_requests_running": "requests_running",
            "vllm:num_requests_waiting": "requests_waiting",
        }
        for line in text.splitlines():
            for metric, key in names.items():
                if line.startswith(metric):
                    with contextlib.suppress(ValueError):
                        stats[key] = stats.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
        return stats
