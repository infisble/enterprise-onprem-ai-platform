from __future__ import annotations

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from sqlalchemy import func, select, text

from apps.api.deps import get_principal, require
from packages.core.cache import get_store
from packages.core.config import get_settings
from packages.core.db import session_scope
from packages.core.models import AuditLog, BatchJob, Document, RequestTrace
from packages.core.security import ACCESS_LEVELS, Permission, Principal
from packages.observability.metrics import KV_CACHE, WINDOW
from services.agent.analytics_sql import run_analytics_query
from services.batch.jobs import QUEUE
from services.llm.embeddings import get_embedder
from services.llm.factory import get_llm
from services.rag.pipeline import ingest_file
from services.search import vector
from services.search.lexical import get_lexical_index
from services.search.rerank import get_reranker

router = APIRouter(tags=["system"])


def _iso(value: datetime) -> str:
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).isoformat()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/health/ready")
def ready() -> dict:
    s = get_settings()
    checks: dict[str, dict] = {}
    try:
        with session_scope() as db:
            db.execute(text("SELECT 1"))
        checks["sql"] = {
            "ok": True,
            "backend": "sqlite" if s.is_sqlite else s.database_url.split("+")[0].split(":")[0],
        }
    except Exception as exc:
        checks["sql"] = {"ok": False, "error": str(exc)[:200]}
    try:
        checks["vector"] = {
            "ok": True,
            "backend": vector.qdrant_backend(),
            "products": vector.count(s.products_collection),
            "document_chunks": vector.count(s.documents_collection),
        }
    except Exception as exc:
        checks["vector"] = {"ok": False, "error": str(exc)[:200]}
    lex = get_lexical_index()
    checks["lexical"] = {"ok": lex.ping(), "backend": lex.backend, "documents": lex.count()}
    store = get_store()
    checks["cache"] = {"ok": store.ping(), "backend": store.backend}
    checks["llm"] = get_llm().health()
    ok = all(c.get("ok") for c in checks.values())
    return {"status": "ready" if ok else "degraded", "checks": checks}


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/api/system/info")
def system_info(principal: Principal = Depends(get_principal)) -> dict:
    s = get_settings()
    return {
        "llm": {
            "provider": s.llm_provider,
            "model": s.active_model,
            "routing": "llm" if s.llm_routing else "rules",
        },
        "embeddings": get_embedder().name,
        "reranker": get_reranker().name,
        "lexical_backend": get_lexical_index().backend,
        "vector_backend": vector.qdrant_backend(),
        "cache_backend": get_store().backend,
        "batch_mode": s.batch_mode,
        "env": s.app_env,
    }


@router.get("/api/metrics/summary")
def metrics_summary(principal: Principal = Depends(get_principal)) -> dict:
    llm = get_llm()
    gpu: dict = {}
    if hasattr(llm, "gpu_stats"):
        gpu = llm.gpu_stats()  # vLLM KV-cache usage / queue depth
        if "kv_cache_usage" in gpu:
            KV_CACHE.set(gpu["kv_cache_usage"])
    with session_scope() as db:
        jobs = db.scalars(select(BatchJob).order_by(BatchJob.id.desc()).limit(20)).all()
        processed = sum(j.processed for j in jobs)
        succeeded = sum(j.succeeded + j.skipped for j in jobs)
        security = dict(db.execute(select(AuditLog.outcome, func.count()).group_by(AuditLog.outcome)).all())
        routes = dict(db.execute(select(RequestTrace.route, func.count()).group_by(RequestTrace.route)).all())
    return {
        "window": WINDOW.summary(),
        "batch": {
            "queue_size": get_store().queue_size(QUEUE),
            "jobs": len(jobs),
            "items_processed": processed,
            "success_rate": round(succeeded / processed, 3) if processed else None,
            "running": sum(1 for j in jobs if j.status == "running"),
        },
        "audit_outcomes": security,
        "routes_all_time": routes,
        "gpu": gpu
        or {"note": "GPU metrics come from vLLM /metrics and NVIDIA DCGM exporter when LLM_PROVIDER=vllm"},
        "gpu_sources": {
            "kv_cache_and_queue": "vLLM /metrics (scraped by the API and by Prometheus)",
            "utilization_and_memory": "NVIDIA DCGM exporter (DCGM_FI_DEV_GPU_UTIL, DCGM_FI_DEV_FB_USED) in Prometheus",
        },
    }


@router.get("/api/traces")
def traces(limit: int = 50, principal: Principal = Depends(get_principal)) -> list[dict]:
    with session_scope() as db:
        q = select(RequestTrace).order_by(RequestTrace.id.desc()).limit(min(limit, 200))
        if not principal.can(Permission.TRACES_READ_ALL):
            q = q.where(RequestTrace.username == principal.username)
        return [
            {
                "request_id": t.request_id,
                "ts": _iso(t.ts),
                "user": t.username,
                "role": t.role,
                "kind": t.kind,
                "query": t.query[:200],
                "route": t.route,
                "llm_provider": t.llm_provider,
                "model": t.model,
                "total_ms": t.total_ms,
                "input_tokens": t.input_tokens,
                "output_tokens": t.output_tokens,
                "tools": [c["tool"] + ":" + c["outcome"] for c in t.tool_calls],
                "error": t.error,
            }
            for t in db.scalars(q)
        ]


@router.get("/api/traces/{request_id}")
def trace_detail(request_id: str, principal: Principal = Depends(get_principal)) -> dict:
    with session_scope() as db:
        t = db.scalar(select(RequestTrace).where(RequestTrace.request_id == request_id))
        if t is None or (t.username != principal.username and not principal.can(Permission.TRACES_READ_ALL)):
            raise HTTPException(404, "trace not found")
        return {
            "request_id": t.request_id,
            "ts": _iso(t.ts),
            "user": t.username,
            "role": t.role,
            "query": t.query,
            "route": t.route,
            "llm_provider": t.llm_provider,
            "model": t.model,
            "total_ms": t.total_ms,
            "input_tokens": t.input_tokens,
            "output_tokens": t.output_tokens,
            "spans": t.spans,
            "tool_calls": t.tool_calls,
            "error": t.error,
        }


@router.get("/api/audit")
def audit(limit: int = 100, principal: Principal = Depends(require(Permission.SYSTEM_MANAGE))) -> list[dict]:
    with session_scope() as db:
        return [
            {
                "ts": _iso(a.ts),
                "request_id": a.request_id,
                "user": a.username,
                "role": a.role,
                "action": a.action,
                "resource": a.resource,
                "outcome": a.outcome,
                "details": a.details,
            }
            for a in db.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 500)))
        ]


@router.get("/api/documents")
def documents(principal: Principal = Depends(require(Permission.DOCS_READ))) -> list[dict]:
    with session_scope() as db:
        return [
            {
                "filename": d.filename,
                "title": d.title,
                "doc_type": d.doc_type,
                "classification": d.classification,
                "chunks": d.chunk_count,
                "flagged_chunks": d.flagged_chunks,
                "ingested_at": _iso(d.ingested_at),
            }
            for d in db.scalars(select(Document).order_by(Document.filename))
            if principal.role.value in d.access_roles
        ]


@router.post("/api/documents")
def upload_document(
    file: UploadFile = File(...),
    access: str = Form("all_staff"),
    title: str = Form(""),
    principal: Principal = Depends(require(Permission.DOCUMENTS_INGEST)),
) -> dict:
    if access not in ACCESS_LEVELS:
        raise HTTPException(400, f"access must be one of {list(ACCESS_LEVELS)}")
    name = Path(file.filename or "upload.txt").name
    tmp = Path(tempfile.mkdtemp()) / name
    with tmp.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        return ingest_file(tmp, {"access": access, "title": title or name, "classification": "internal"})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class SQLIn(BaseModel):
    sql: str


@router.post("/api/analytics/sql")
def analytics(body: SQLIn, principal: Principal = Depends(require(Permission.ANALYTICS_SQL))) -> dict:
    return run_analytics_query(body.sql, principal)
