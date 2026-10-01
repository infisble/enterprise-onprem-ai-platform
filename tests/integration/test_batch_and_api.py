from fastapi.testclient import TestClient
from sqlalchemy import select

from packages.core.db import session_scope
from packages.core.models import BatchItem, BatchJob, Product
from services.batch.jobs import create_job, resume_job, run_job


def _job(job_id):
    with session_scope() as s:
        return s.get(BatchJob, job_id)


def test_batch_fills_missing_content_and_is_idempotent():
    job_id = create_job(limit=15, created_by="test")
    assert run_job(job_id) == "completed"
    job = _job(job_id)
    assert job.processed == 15 and job.failed == 0
    with session_scope() as s:
        items = s.scalars(select(BatchItem).where(BatchItem.job_id == job_id)).all()
        ids = [i.product_id for i in items]
        products = s.scalars(select(Product).where(Product.id.in_(ids))).all()
        assert all(p.description_sk and p.description_cs for p in products)
        filled = [p for p in products if p.content_source == "ai"]
        assert filled and all(p.needs_review for p in filled)

    # Second run over the same products: nothing left to generate -> only cheap checks, no duplicate LLM work
    job2 = create_job(limit=15, created_by="test", only_missing=False)
    assert run_job(job2) == "completed"
    with session_scope() as s:
        actions = [a for i in s.scalars(select(BatchItem).where(BatchItem.job_id == job2)) for a in i.actions]
    assert not any(a.startswith(("generate_sk:filled", "translate_cs:filled")) for a in actions)
    assert any("skipped(idempotent)" in a for a in actions)


def test_batch_retries_failures_and_resume():
    job_id = create_job(limit=30, created_by="test", fail_rate=0.5)
    status = run_job(job_id)
    job = _job(job_id)
    assert job.processed == 30
    if status == "completed_with_errors":
        assert job.failed > 0
        with session_scope() as s:
            failed = s.scalars(
                select(BatchItem).where(BatchItem.job_id == job_id, BatchItem.status == "failed")
            ).all()
            assert all(i.attempts == 3 and "simulated" in i.last_error for i in failed), [
                (i.attempts, i.last_error) for i in failed
            ]
            s.get(BatchJob, job_id).params = {**job.params, "fail_rate": 0.0}
        resume_job(job_id)
        assert run_job(job_id) == "completed"
        assert _job(job_id).failed == 0


def _client():
    from apps.api.main import app

    return TestClient(app)


def _login(client, username):
    r = client.post("/api/auth/login", json={"username": username, "password": "demo"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_api_auth_and_rbac():
    with _client() as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
        assert (
            client.post("/api/auth/login", json={"username": "eva.employee", "password": "x"}).status_code
            == 401
        )
        emp = _login(client, "eva.employee")
        assert client.get("/api/batch/jobs", headers=emp).status_code == 403
        assert client.post("/api/analytics/sql", json={"sql": "SELECT 1"}, headers=emp).status_code == 403
        product = client.get("/api/products/NP-10042", headers=emp).json()
        assert product["stock"] is None and product["pricing"] is None
        adm = _login(client, "adam.admin")
        assert client.get("/api/products/NP-10042", headers=adm).json()["stock"]["total_available"] == 1620


def test_api_chat_search_metrics():
    with _client() as client:
        h = _login(client, "samuel.sales")
        r = client.post(
            "/api/chat",
            json={"message": "How many pieces of SKU NP-10042 are currently available?"},
            headers=h,
        )
        assert r.status_code == 200 and r.json()["route"] == "stock"
        assert r.headers.get("x-request-id")
        s = client.get(
            "/api/search/products", params={"q": "black bottle", "price_max": 10}, headers=h
        ).json()
        assert s["results"] and all(x["price_eur"] <= 10 for x in s["results"])
        assert client.get("/health/ready").json()["status"] == "ready"
        assert "http_requests_total" in client.get("/metrics").text
        traces = client.get("/api/traces", headers=h).json()
        assert traces and all(t["user"] == "samuel.sales" for t in traces)
        assert client.get("/api/metrics/summary", headers=h).json()["window"]["requests"] >= 1


def test_stale_running_job_is_recovered_after_worker_crash():
    from datetime import UTC, datetime, timedelta

    from services.batch.jobs import recover_stale_jobs

    job_id = create_job(limit=3, created_by="test")
    with session_scope() as s:  # simulate a worker that was OOM-killed mid-job 10 minutes ago
        job = s.get(BatchJob, job_id)
        job.status = "running"
        job.stats = {"heartbeat": (datetime.now(UTC) - timedelta(minutes=10)).isoformat()}
    assert job_id in recover_stale_jobs()
    assert _job(job_id).status == "queued"
    assert run_job(job_id) == "completed"


def test_fresh_running_job_cannot_be_resumed():
    from datetime import UTC, datetime

    import pytest

    job_id = create_job(limit=2, created_by="test")
    with session_scope() as s:
        job = s.get(BatchJob, job_id)
        job.status = "running"
        job.stats = {"heartbeat": datetime.now(UTC).isoformat()}
    with pytest.raises(ValueError):
        resume_job(job_id)


def test_transient_error_classification():
    import httpx

    from services.batch.jobs import TransientError, is_transient

    req = httpx.Request("POST", "http://gpu/v1/chat/completions")
    assert is_transient(TransientError("x"))
    assert is_transient(httpx.ReadTimeout("slow", request=req))
    assert is_transient(httpx.HTTPStatusError("503", request=req, response=httpx.Response(503, request=req)))
    assert is_transient(httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, request=req)))
    assert not is_transient(
        httpx.HTTPStatusError("400", request=req, response=httpx.Response(400, request=req))
    )
    assert not is_transient(ValueError("bad data"))


def test_nightly_guard_is_atomic():
    from packages.core.cache import InMemoryStore

    store = InMemoryStore()
    assert store.set_if_absent("batch:nightly:2026-10-01", True, ttl_s=60)
    assert not store.set_if_absent("batch:nightly:2026-10-01", True, ttl_s=60)
