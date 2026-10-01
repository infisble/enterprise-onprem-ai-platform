"""Explicit tool registry and governed executor.

The agent can only do what is registered here. Each tool declares:
  * a Pydantic input schema (arguments are validated before execution - no free-form strings to SQL)
  * the permission required (checked against the caller's role, not the model's opinion)
  * side_effect: read | write   (write tools never mutate product data directly; they create
    approval requests - publication happens only through the approval workflow)

The executor records every call (args, outcome, latency) in the trace, Prometheus and - for denied or
write calls - the SQL audit log.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from packages.core.db import session_scope
from packages.core.models import AuditLog
from packages.core.security import Permission, Principal
from packages.observability.logging import get_logger
from packages.observability.metrics import SECURITY_EVENTS, TOOL_CALLS
from packages.observability.tracing import current_trace, span
from packages.schemas.chat import ToolCallRecord

log = get_logger(__name__)


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    permission: Permission
    side_effect: str  # read | write
    fn: Callable[[Any, Principal], Any]  # receives the validated input_model instance
    requires_approval: bool = False

    def openai_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def allowed_for(self, principal: Principal) -> list[ToolSpec]:
        """Only tools the caller may use are ever shown to the LLM (least privilege)."""
        return [t for t in self._tools.values() if principal.can(t.permission)]

    def execute(self, name: str, args: dict[str, Any], principal: Principal) -> tuple[Any, ToolCallRecord]:
        t0 = time.perf_counter()
        spec = self._tools.get(name)
        trace = current_trace()

        def record(outcome: str, side: str, preview: str = "", error: str = "") -> ToolCallRecord:
            rec = ToolCallRecord(
                tool=name,
                args=args,
                outcome=outcome,
                side_effect=side,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                result_preview=preview[:300],
                error=error,
            )
            TOOL_CALLS.labels(tool=name, outcome=outcome).inc()
            if trace is not None:
                trace.tool_calls.append(rec.model_dump())
            return rec

        if spec is None:
            SECURITY_EVENTS.labels(event="unknown_tool").inc()
            self._audit(principal, name, "denied", {"reason": "unknown tool", "args": args})
            return None, record("denied", "unknown", error="Tool is not in the allowlist")
        if not principal.can(spec.permission):
            SECURITY_EVENTS.labels(event="tool_denied").inc()
            self._audit(
                principal, name, "denied", {"reason": f"missing {spec.permission.value}", "args": args}
            )
            return None, record(
                "denied",
                spec.side_effect,
                error=f"Role '{principal.role.value}' is not allowed to use {name} "
                f"(requires {spec.permission.value})",
            )
        try:
            parsed = spec.input_model.model_validate(args)
        except ValidationError as exc:
            return None, record("invalid_args", spec.side_effect, error=exc.errors()[0]["msg"])
        try:
            with span(name, kind="tool", side_effect=spec.side_effect):
                result = spec.fn(parsed, principal)
        except ToolError as exc:
            return None, record("error", spec.side_effect, error=str(exc))
        except Exception as exc:  # unexpected: log with stack, do not leak internals to the user
            log.exception("tool failed", extra={"tool": name})
            return None, record(
                "error", spec.side_effect, error=f"internal error in {name}: {type(exc).__name__}"
            )
        if spec.side_effect == "write":
            self._audit(principal, name, "success", {"args": args})
        preview = result.model_dump_json() if isinstance(result, BaseModel) else str(result)
        return result, record("success", spec.side_effect, preview)

    @staticmethod
    def _audit(principal: Principal, action: str, outcome: str, details: dict[str, Any]) -> None:
        trace = current_trace()
        with session_scope() as s:
            s.add(
                AuditLog(
                    request_id=trace.request_id if trace else "",
                    username=principal.username,
                    role=principal.role.value,
                    action=f"tool:{action}",
                    resource=str(details.get("args", ""))[:200],
                    outcome=outcome,
                    details=details,
                )
            )
