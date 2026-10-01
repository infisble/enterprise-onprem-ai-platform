from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    ref: int
    source: str
    doc_type: str
    locator: dict[str, Any]  # e.g. {"page": 8, "section": "Returns"} or {"sheet": "Products", "row": 124}
    snippet: str
    score: float

    def label(self) -> str:
        parts = [self.source]
        for key in ("sheet", "row", "section", "page", "path"):
            if key in self.locator and self.locator[key] not in (None, ""):
                parts.append(f"{key.capitalize()}: {self.locator[key]}")
        return " | ".join(parts)


class ToolCallRecord(BaseModel):
    tool: str
    args: dict[str, Any]
    outcome: str  # success | denied | error | invalid_args
    side_effect: str  # read | write
    duration_ms: float
    result_preview: str = ""
    error: str = ""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    route: str
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    products: list[dict[str, Any]] = Field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    security_notices: list[str] = Field(default_factory=list)
    latency: dict[str, Any] = Field(default_factory=dict)
    llm: dict[str, Any] = Field(default_factory=dict)
