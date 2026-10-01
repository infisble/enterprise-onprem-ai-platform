from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from apps.api.deps import get_principal
from packages.core.db import session_scope
from packages.core.models import User
from packages.core.security import DEMO_USERS, Principal, Role, create_token, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(body: LoginIn) -> dict:
    with session_scope() as s:
        user = s.scalar(select(User).where(User.username == body.username, User.active.is_(True)))
        if user is None or not verify_password(body.password, user.password_hash):
            raise HTTPException(401, "Invalid credentials")
        principal = Principal(user.username, Role(user.role), user.full_name)
    return {
        "access_token": create_token(principal),
        "token_type": "bearer",
        "user": {
            "username": principal.username,
            "role": principal.role.value,
            "full_name": principal.full_name,
            "permissions": sorted(p.value for p in principal.permissions),
        },
    }


@router.get("/me")
def me(principal: Principal = Depends(get_principal)) -> dict:
    return {
        "username": principal.username,
        "role": principal.role.value,
        "full_name": principal.full_name,
        "permissions": sorted(p.value for p in principal.permissions),
    }


@router.get("/demo-users")
def demo_users() -> list[dict]:
    """Demo convenience only - disable in production."""
    return DEMO_USERS
