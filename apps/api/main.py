"""FastAPI gateway: authentication, RBAC, request tracing and the HTTP surface of the platform.

A modular monolith on purpose: one deployable API + one worker process with clear internal module
boundaries (search, rag, agent, batch, llm) is what a 25-50 person company can operate. Modules can be
split into services later along these boundaries if scale ever requires it.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.routers import auth, catalog, chat, system, workflow
from packages.core.config import get_settings
from packages.core.db import init_db
from packages.observability.logging import configure_logging, get_logger, request_ctx
from packages.observability.metrics import ERRORS, HTTP_LATENCY, HTTP_REQUESTS

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    init_db()
    from services.search.hybrid import get_search_service

    get_search_service().ensure_lexical_loaded()
    worker = None
    if s.batch_mode == "inline":
        from services.batch.jobs import recover_stale_jobs
        from services.batch.worker import Worker

        recover_stale_jobs()
        worker = Worker()
        worker.start_background()
    log.info(
        "api started",
        extra={"llm_provider": s.llm_provider, "model": s.active_model, "batch_mode": s.batch_mode},
    )
    yield
    if worker:
        worker.stop()


app = FastAPI(
    title="NovaPromo Enterprise AI Platform",
    version="0.1.0",
    description="On-prem AI platform showcase: hybrid search, permission-aware RAG, governed agent tools, "
    "human approvals, batch enrichment and observability.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = request_ctx.set({"request_id": request_id})
    route = request.url.path
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    except Exception:
        ERRORS.labels(component="api").inc()
        log.exception("unhandled error", extra={"path": route})
        return JSONResponse({"detail": "internal error", "request_id": request_id}, status_code=500)
    finally:
        matched = request.scope.get("route")
        label = getattr(matched, "path", route) if matched else "unmatched"
        elapsed = time.perf_counter() - t0
        HTTP_REQUESTS.labels(request.method, label, str(status)).inc()
        HTTP_LATENCY.labels(request.method, label).observe(elapsed)
        if label not in ("/metrics", "/health"):
            log.info(
                "http",
                extra={
                    "method": request.method,
                    "path": label,
                    "status": status,
                    "ms": round(elapsed * 1000, 1),
                },
            )
        request_ctx.reset(token)


for r in (auth.router, chat.router, catalog.router, workflow.router, system.router):
    app.include_router(r)
