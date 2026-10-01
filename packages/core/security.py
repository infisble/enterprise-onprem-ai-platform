"""Authentication (JWT) and role-based access control.

RBAC is enforced in three places, never in the prompt:
1. API layer   – endpoint dependencies (`require_permission`).
2. Tool layer  – the tool executor checks the principal before running any tool.
3. Data layer  – RAG filters chunks by `access_roles` in the Qdrant query *and* re-checks
                 before context assembly, so forbidden text never reaches the LLM.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import jwt

from packages.core.config import get_settings


class Role(StrEnum):
    EMPLOYEE = "employee"
    SALES = "sales"
    CATALOG_MANAGER = "catalog_manager"
    ADMIN = "admin"


class Permission(StrEnum):
    DOCS_READ = "docs:read"
    PRODUCTS_SEARCH = "products:search"
    PRODUCTS_READ = "products:read"
    STOCK_READ = "stock:read"
    PRICING_READ = "pricing:read"  # volume/tier pricing (list price is public)
    SUPPLIERS_READ = "suppliers:read"
    CONTENT_GENERATE = "content:generate"
    UPDATES_REQUEST = "updates:request"
    UPDATES_APPROVE = "updates:approve"
    BATCH_READ = "batch:read"
    BATCH_RUN = "batch:run"
    ANALYTICS_SQL = "analytics:sql"
    TRACES_READ_ALL = "traces:read_all"
    DOCUMENTS_INGEST = "documents:ingest"
    SYSTEM_MANAGE = "system:manage"


_BASE = {Permission.DOCS_READ, Permission.PRODUCTS_SEARCH, Permission.PRODUCTS_READ}
_SALES = _BASE | {Permission.STOCK_READ, Permission.PRICING_READ, Permission.SUPPLIERS_READ}
_CATALOG = _SALES | {
    Permission.CONTENT_GENERATE,
    Permission.UPDATES_REQUEST,
    Permission.BATCH_READ,
}

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.EMPLOYEE: frozenset(_BASE),
    Role.SALES: frozenset(_SALES),
    Role.CATALOG_MANAGER: frozenset(_CATALOG),
    Role.ADMIN: frozenset(Permission),
}

# Document access levels -> roles allowed to retrieve chunks of such documents.
ACCESS_LEVELS: dict[str, list[str]] = {
    "all_staff": [r.value for r in Role],
    "commercial": [Role.SALES.value, Role.CATALOG_MANAGER.value, Role.ADMIN.value],
    "catalog": [Role.CATALOG_MANAGER.value, Role.ADMIN.value],
    "management": [Role.ADMIN.value],
}


@dataclass(frozen=True)
class Principal:
    username: str
    role: Role
    full_name: str = ""

    @property
    def permissions(self) -> frozenset[Permission]:
        return ROLE_PERMISSIONS[self.role]

    def can(self, permission: Permission) -> bool:
        return permission in self.permissions


class AuthorizationError(Exception):
    def __init__(self, principal: Principal, permission: Permission | str, resource: str = ""):
        self.principal = principal
        self.permission = str(permission)
        self.resource = resource
        super().__init__(
            f"Role '{principal.role.value}' lacks permission '{self.permission}'"
            + (f" for {resource}" if resource else "")
        )


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return f"pbkdf2_sha256${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 120_000)
    return hmac.compare_digest(candidate.hex(), digest_hex)


def create_token(principal: Principal) -> str:
    s = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": principal.username,
        "role": principal.role.value,
        "name": principal.full_name,
        "iat": now,
        "exp": now + timedelta(minutes=s.jwt_ttl_minutes),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> Principal:
    payload = jwt.decode(token, get_settings().jwt_secret, algorithms=["HS256"])
    return Principal(username=payload["sub"], role=Role(payload["role"]), full_name=payload.get("name", ""))


DEMO_USERS: list[dict[str, str]] = [
    {"username": "eva.employee", "full_name": "Eva Horváthová", "role": Role.EMPLOYEE.value},
    {"username": "samuel.sales", "full_name": "Samuel Kováč", "role": Role.SALES.value},
    {"username": "katarina.catalog", "full_name": "Katarína Nováková", "role": Role.CATALOG_MANAGER.value},
    {"username": "adam.admin", "full_name": "Adam Baláž", "role": Role.ADMIN.value},
]
