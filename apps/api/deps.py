from __future__ import annotations

from collections.abc import Callable

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from packages.core.db import session_scope
from packages.core.models import AuditLog
from packages.core.security import Permission, Principal, decode_token
from packages.observability.logging import request_ctx
from packages.observability.metrics import SECURITY_EVENTS

bearer = HTTPBearer(auto_error=False)


def get_principal(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Principal:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        principal = decode_token(creds.credentials)
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {exc}") from exc
    ctx = dict(request_ctx.get())
    ctx.update(user_id=principal.username, role=principal.role.value)
    request_ctx.set(ctx)
    return principal


def require(permission: Permission) -> Callable[[Principal], Principal]:
    def dep(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.can(permission):
            SECURITY_EVENTS.labels(event="api_denied").inc()
            with session_scope() as s:
                s.add(
                    AuditLog(
                        request_id=request_ctx.get().get("request_id", ""),
                        username=principal.username,
                        role=principal.role.value,
                        action=f"api:{permission.value}",
                        outcome="denied",
                        details={},
                    )
                )
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Role '{principal.role.value}' lacks permission '{permission.value}'",
            )
        return principal

    return dep
