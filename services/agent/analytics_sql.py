"""Controlled analytics-SQL mode (admin only, off the default agent path).

The agent's default data access is through narrow business tools (`get_product_stock(sku)`), never
`execute_any_sql(query)`. For ad-hoc analytics an admin may run SQL that passes ALL of these gates:

  1. single statement, parsed into an AST with sqlglot (no regex guessing)
  2. only SELECT / UNION of SELECTs - any DML/DDL/command node anywhere in the tree is rejected
  3. table allowlist (no customers, users, audit_log ...), no system schemas, no dangerous functions
  4. LIMIT enforced / capped at ANALYTICS_ROW_LIMIT
  5. executed with a *read-only database role* (PostgreSQL `analytics_ro`, SQLite `mode=ro`)
  6. statement timeout (PostgreSQL `statement_timeout`, SQLite progress-handler abort)
  7. audit-logged with the normalised SQL, row count and outcome
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlglot import exp

from packages.core.config import get_settings
from packages.core.db import session_scope
from packages.core.models import AuditLog
from packages.core.security import Principal
from packages.observability.tracing import current_trace, span

ALLOWED_TABLES = {"products", "stock", "orders", "order_items", "suppliers", "price_tiers"}
FORBIDDEN_FUNCTIONS = {
    "pg_sleep",
    "pg_read_file",
    "pg_ls_dir",
    "lo_import",
    "dblink",
    "load_extension",
    "readfile",
    "writefile",
    "xp_cmdshell",
    "openrowset",
    "sleep",
    "benchmark",
}
FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
    exp.Merge,
    exp.TruncateTable,
    exp.Grant,
    exp.Into,
)


class SQLValidationError(ValueError):
    pass


@dataclass
class ValidatedQuery:
    sql: str
    tables: list[str]
    limit: int


def validate_sql(sql: str, dialect: str = "postgres", row_limit: int | None = None) -> ValidatedQuery:
    row_limit = row_limit or get_settings().analytics_row_limit
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise SQLValidationError(f"could not parse SQL: {str(exc).splitlines()[0]}") from exc
    if len(statements) != 1:
        raise SQLValidationError("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Select | exp.Union | exp.Intersect | exp.Except):
        raise SQLValidationError(f"only SELECT queries are allowed (got {tree.key.upper()})")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SQLValidationError(f"forbidden operation: {node.key.upper()}")
        if isinstance(node, exp.Anonymous | exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in FORBIDDEN_FUNCTIONS:
                raise SQLValidationError(f"forbidden function: {name}")
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    tables = sorted({t.name.lower() for t in tree.find_all(exp.Table)} - cte_names)
    for t in tree.find_all(exp.Table):
        if t.db and t.db.lower() in {"information_schema", "pg_catalog", "sys", "sqlite_master"}:
            raise SQLValidationError(f"system schema access is not allowed: {t.db}")
    disallowed = [t for t in tables if t not in ALLOWED_TABLES]
    if disallowed:
        raise SQLValidationError(f"table(s) not in analytics allowlist: {', '.join(disallowed)}")
    limit_node = tree.args.get("limit")
    current = None
    if limit_node is not None:
        try:
            current = int(limit_node.expression.name)
        except (AttributeError, ValueError):
            current = None
    effective = min(current or row_limit, row_limit)
    tree = (
        tree.limit(effective)
        if isinstance(tree, exp.Select)
        else exp.select("*").from_(tree.subquery("q")).limit(effective)
    )
    return ValidatedQuery(sql=tree.sql(dialect=dialect), tables=tables, limit=effective)


_engine: Engine | None = None


def _analytics_engine() -> Engine:
    global _engine
    if _engine is None:
        s = get_settings()
        url = s.effective_analytics_url
        if url.startswith("sqlite:///"):
            path = url.replace("sqlite:///", "")
            _engine = create_engine(
                f"sqlite:///file:{path}?mode=ro&uri=true", connect_args={"check_same_thread": False}
            )
        else:
            _engine = create_engine(url, pool_pre_ping=True, pool_size=2)
    return _engine


def run_analytics_query(sql: str, principal: Principal) -> dict[str, Any]:
    s = get_settings()
    engine = _analytics_engine()
    dialect = {"sqlite": "sqlite", "mssql": "tsql"}.get(engine.dialect.name, "postgres")
    trace = current_trace()
    outcome, error, rows, columns = "success", "", [], []
    validated: ValidatedQuery | None = None
    t0 = time.perf_counter()
    try:
        validated = validate_sql(sql, dialect=dialect)
        with span("analytics_sql", kind="sql", tables=validated.tables), engine.connect() as conn:
            if dialect == "postgres":
                conn.execute(text(f"SET statement_timeout = {int(s.analytics_timeout_ms)}"))
                conn.execute(text("SET default_transaction_read_only = on"))
            elif dialect == "tsql":
                # Read-only is enforced by the login (db_datareader on allowlisted tables only);
                # pyodbc's connection timeout bounds query execution time.
                conn.connection.driver_connection.timeout = max(1, int(s.analytics_timeout_ms / 1000))  # type: ignore[union-attr]
            else:
                deadline = time.perf_counter() + s.analytics_timeout_ms / 1000
                raw = conn.connection.driver_connection
                raw.set_progress_handler(  # type: ignore[union-attr]
                    lambda: 1 if time.perf_counter() > deadline else 0, 10_000
                )
            result = conn.execute(text(validated.sql))
            columns = list(result.keys())
            rows = [list(r) for r in result.fetchmany(validated.limit)]
    except SQLValidationError as exc:
        outcome, error = "denied", str(exc)
    except Exception as exc:
        outcome, error = "error", f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    finally:
        with session_scope() as db:
            db.add(
                AuditLog(
                    request_id=trace.request_id if trace else "",
                    username=principal.username,
                    role=principal.role.value,
                    action="analytics_sql",
                    resource=sql[:200],
                    outcome=outcome,
                    details={
                        "validated_sql": validated.sql if validated else None,
                        "rows": len(rows),
                        "error": error,
                        "ms": round((time.perf_counter() - t0) * 1000, 1),
                    },
                )
            )
    return {
        "outcome": outcome,
        "error": error,
        "sql": validated.sql if validated else sql,
        "columns": columns,
        "rows": rows[: validated.limit if validated else 0],
        "row_count": len(rows),
    }


def apply_readonly_grants(role: str = "analytics_ro") -> str:
    """Defence in depth for PostgreSQL: the analytics role can SELECT only the allowlisted tables,
    so even a validator bug could not read customers, users or the audit log."""
    from packages.core.db import get_engine

    engine = get_engine()
    if engine.dialect.name != "postgresql":
        return "skipped (not PostgreSQL)"
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role}).scalar()
        if not exists:
            return f"skipped (role {role} does not exist)"
        conn.execute(text(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {role}"))
        for table in sorted(ALLOWED_TABLES):
            conn.execute(text(f"GRANT SELECT ON {table} TO {role}"))
    return f"{role}: SELECT on {', '.join(sorted(ALLOWED_TABLES))}"


def reset_analytics_engine() -> None:
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None
