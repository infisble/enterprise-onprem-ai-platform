"""Test harness: an isolated, fully bootstrapped platform (temp SQLite + in-memory Qdrant + BM25 + mock LLM).

Environment is configured BEFORE any application module reads settings.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="novapromo-test-"))
os.environ.update(
    {
        "APP_ENV": "test",
        "DATABASE_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
        "ANALYTICS_DATABASE_URL": "",
        "QDRANT_URL": "",
        "QDRANT_LOCAL_PATH": ":memory:",
        "ELASTICSEARCH_URL": "",
        "REDIS_URL": "",
        "LLM_PROVIDER": "mock",
        "EMBEDDING_PROVIDER": "hash",
        "RERANKER_PROVIDER": "feature",
        "BATCH_MODE": "inline",
        "LOG_JSON": "false",
        "LOG_LEVEL": "WARNING",
        "JWT_SECRET": "test-secret-test-secret-test-secret",
    }
)

from packages.core.config import get_settings, reset_settings_cache  # noqa: E402

reset_settings_cache()
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def platform():
    from packages.core.db import init_db, session_scope
    from services.ingestion.loader import seed_customers_and_orders, seed_users, upsert_products
    from services.ingestion.suppliers import load_all_feeds
    from services.rag.pipeline import ensure_documents_collection, ingest_directory
    from services.search.hybrid import get_search_service

    assert get_settings().app_env == "test"
    init_db()
    data = ROOT / "data"
    with session_scope() as s:
        seed_users(s)
    report = load_all_feeds(data / "suppliers")
    with session_scope() as s:
        upsert_products(s, report, data)
    with session_scope() as s:
        seed_customers_and_orders(s, n_customers=30, n_orders=120)
    get_search_service().index_all(recreate=True)
    ensure_documents_collection(recreate=True)
    ingest_directory(data / "documents", force=True)
    yield {"report": report}


def _principal(username: str):
    from packages.core.security import DEMO_USERS, Principal, Role

    u = next(x for x in DEMO_USERS if x["username"] == username)
    return Principal(u["username"], Role(u["role"]), u["full_name"])


@pytest.fixture
def employee():
    return _principal("eva.employee")


@pytest.fixture
def sales():
    return _principal("samuel.sales")


@pytest.fixture
def catalog():
    return _principal("katarina.catalog")


@pytest.fixture
def admin():
    return _principal("adam.admin")


@pytest.fixture
def llm_spy(monkeypatch):
    """Capture every prompt/payload sent to the LLM (to assert what the model can and cannot see)."""
    from services.llm.factory import get_llm

    llm = get_llm()
    calls: list[dict] = []
    original = llm._complete

    def spy(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        return original(messages, **kwargs)

    monkeypatch.setattr(llm, "_complete", spy)
    return calls
