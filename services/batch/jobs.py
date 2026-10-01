"""Nightly product-enrichment batch.

    Scheduler ─► job row (SQL) + item rows ─► queue (Redis) ─► Worker ─► thread pool ─► per-item tasks
                                                                          │
               validate ► generate SK ► translate SK→CS ► classify-check ► embed + index ► record
                                                                          │
                          ledger (idempotency) · retries w/ backoff · checkpoint · pause / resume

* Idempotency: each (product, task, input_hash) is recorded in `processing_ledger`. Re-running a job,
  resuming after a crash, or a second nightly run over unchanged data performs no duplicate LLM calls.
* Checkpointing: job counters + last processed product id are committed every N items; a resumed job
  continues with the remaining pending/failed items only.
* Content policy: AI may fill EMPTY fields (marked content_source=ai, needs_review=true); overwriting
  existing content always goes through an approval request.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, or_, select, update

from packages.core.cache import get_store
from packages.core.config import get_settings
from packages.core.db import session_scope
from packages.core.models import BatchItem, BatchJob, ProcessingLedger, Product
from packages.core.repository import available_stock, supplier_codes
from packages.core.text import COLOR_LOOKUP
from packages.observability.logging import get_logger
from packages.observability.metrics import BATCH_ITEMS, BATCH_QUEUE, BATCH_RUNNING
from services.agent.validation import validate_description
from services.llm.factory import get_llm
from services.llm.prompts import (
    PROMPT_VERSION,
    classification_messages,
    description_messages,
    translation_messages,
)
from services.search.documents import product_search_doc

log = get_logger(__name__)
QUEUE = "batch:jobs"
CATEGORIES = ["drinkware", "apparel", "bags", "writing", "office", "technology", "umbrellas", "lifestyle"]
GUIDELINES = [
    "350-600 characters",
    "no exclamation marks or superlatives",
    "mention material and colour",
    "end with branding options",
]


class TransientError(Exception):
    """Retryable failure (LLM timeout, 503 from the GPU server, ...)."""


def is_transient(exc: Exception) -> bool:
    """Timeouts, connection errors, HTTP 429 and 5xx from the inference server are worth retrying."""
    import httpx

    if isinstance(exc, TransientError | httpx.TimeoutException | httpx.TransportError | ConnectionError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


def _hash(*parts: Any) -> str:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()


# ------------------------------------------------------------------------------------ jobs
def create_job(
    job_type: str = "nightly_enrichment",
    created_by: str = "scheduler",
    limit: int | None = None,
    only_missing: bool = True,
    fail_rate: float = 0.0,
) -> int:
    params = {
        "limit": limit,
        "only_missing": only_missing,
        "fail_rate": min(max(fail_rate, 0.0), 0.5),
        "prompt_version": PROMPT_VERSION,
    }
    with session_scope() as s:
        q = select(Product.id).where(Product.status == "active").order_by(Product.id)
        if only_missing:
            q = q.where(or_(Product.description_sk == "", Product.description_cs == ""))
        if limit:
            q = q.limit(limit)
        ids = list(s.scalars(q))
        job = BatchJob(
            job_type=job_type, status="queued", params=params, created_by=created_by, total=len(ids)
        )
        s.add(job)
        s.flush()
        s.bulk_save_objects([BatchItem(job_id=job.id, product_id=pid) for pid in ids])
        job_id = job.id
    get_store().push(QUEUE, {"job_id": job_id})
    BATCH_QUEUE.set(get_store().queue_size(QUEUE))
    log.info("batch job created", extra={"job_id": job_id, "items": len(ids), "created_by": created_by})
    return job_id


def request_pause(job_id: int) -> None:
    get_store().set_json(f"batch:pause:{job_id}", True, ttl_s=3600)


STALE_AFTER_S = 300  # a running job without a checkpoint for 5 min is assumed dead (worker crash / OOM kill)


def _heartbeat_age_s(job: BatchJob) -> float:
    beat = (job.stats or {}).get("heartbeat") or (job.started_at.isoformat() if job.started_at else None)
    if not beat:
        return float("inf")
    ts = datetime.fromisoformat(beat)
    return (datetime.now(UTC) - (ts if ts.tzinfo else ts.replace(tzinfo=UTC))).total_seconds()


def recover_stale_jobs() -> list[int]:
    """Called on worker start: re-queue jobs left 'running' by a crashed worker."""
    with session_scope() as s:
        stale = [
            j.id
            for j in s.scalars(select(BatchJob).where(BatchJob.status == "running"))
            if _heartbeat_age_s(j) > STALE_AFTER_S
        ]
    for job_id in stale:
        log.warning("recovering stale batch job", extra={"job_id": job_id})
        resume_job(job_id)
    return stale


def resume_job(job_id: int) -> None:
    with session_scope() as s:
        job = s.get(BatchJob, job_id)
        if job is None:
            raise ValueError("job not found")
        stale = job.status == "running" and _heartbeat_age_s(job) > STALE_AFTER_S
        if job.status not in ("paused", "failed", "completed_with_errors") and not stale:
            raise ValueError("only paused, failed, partially failed or stale running jobs can be resumed")
        # failed items get a fresh retry budget on resume; counters are rolled back accordingly
        reset = (
            s.execute(
                update(BatchItem)
                .where(BatchItem.job_id == job_id, BatchItem.status == "failed")
                .values(status="pending", attempts=0)
            ).rowcount  # type: ignore[attr-defined]
            or 0
        )
        job.failed = max(0, job.failed - reset)
        job.processed = max(0, job.processed - reset)
        job.status = "queued"
    get_store().set_json(f"batch:pause:{job_id}", False, ttl_s=60)
    get_store().push(QUEUE, {"job_id": job_id})


# ------------------------------------------------------------------------------------ tasks
class ItemProcessor:
    def __init__(self, job_id: int, fail_rate: float) -> None:
        self.job_id = job_id
        self.fail_rate = fail_rate
        self.llm = get_llm()
        self.settings = get_settings()
        self.tokens = 0
        self._lock = threading.Lock()

    def _llm(self, messages, task: str, payload: dict) -> str:
        if self.fail_rate and random.random() < self.fail_rate:
            raise TransientError(f"simulated inference timeout during {task}")
        res = self.llm.complete(messages, task=task, payload=payload)
        with self._lock:
            self.tokens += res.input_tokens + res.output_tokens
        return res.text.strip()

    def _done(self, s, product_id: int, task: str, h: str) -> bool:
        return (
            s.scalar(
                select(ProcessingLedger.id).where(
                    ProcessingLedger.product_id == product_id,
                    ProcessingLedger.task == task,
                    ProcessingLedger.input_hash == h,
                )
            )
            is not None
        )

    def _record(self, s, product_id: int, task: str, h: str) -> None:
        s.add(ProcessingLedger(product_id=product_id, task=task, input_hash=h, job_id=self.job_id))

    def process(self, product_id: int) -> list[str]:
        actions: list[str] = []
        with session_scope() as s:
            p = s.get(Product, product_id)
            if p is None:
                return ["missing"]
            facts = {
                "sku": p.sku,
                "name_sk": p.name_sk,
                "name_cs": p.name_cs,
                "name_en": p.name_en,
                "category": p.category,
                "subcategory": p.subcategory,
                "material": p.material,
                "color": p.color,
                "capacity_ml": p.capacity_ml,
                "sizes": p.sizes or [],
                "tags": p.tags or [],
            }
            desc_sk, desc_cs = p.description_sk, p.description_cs

        # 1. validate master data
        issues = []
        if facts["color"] not in COLOR_LOOKUP.values():
            issues.append("unknown colour")
        if not facts["name_sk"] or facts["name_sk"] == facts["name_en"]:
            issues.append("missing Slovak name")
        actions.append("validate:" + ("ok" if not issues else ";".join(issues)))

        # 2. generate missing Slovak description
        if not desc_sk:
            h = _hash("generate_sk", facts, PROMPT_VERSION)
            with session_scope() as s:
                done = self._done(s, product_id, "generate_sk", h)
            if done:
                actions.append("generate_sk:skipped(idempotent)")
            else:
                text = self._llm(
                    description_messages(facts, GUIDELINES, "sk"),
                    "product_description",
                    {"product": facts, "language": "sk"},
                )
                val = validate_description(text, "sk", facts)
                if val["valid"] or not val["errors"]:
                    self._fill(product_id, "description_sk", text, h, "generate_sk")
                    desc_sk = text
                    actions.append("generate_sk:filled")
                else:
                    actions.append("generate_sk:rejected(" + "; ".join(val["errors"]) + ")")

        # 3. translate SK -> CS
        if not desc_cs and desc_sk:
            h = _hash("translate_cs", desc_sk, PROMPT_VERSION)
            with session_scope() as s:
                done = self._done(s, product_id, "translate_cs", h)
            if done:
                actions.append("translate_cs:skipped(idempotent)")
            else:
                text = self._llm(
                    translation_messages(desc_sk, "sk", "cs"),
                    "translate",
                    {"text": desc_sk, "source": "sk", "target": "cs"},
                )
                self._fill(product_id, "description_cs", text, h, "translate_cs")
                actions.append("translate_cs:filled")

        # 4. classification consistency check (flags, never overwrites the category)
        h = _hash("classify", facts["name_en"], facts["tags"], PROMPT_VERSION)
        with session_scope() as s:
            done = self._done(s, product_id, "classify", h)
        if not done:
            raw = self._llm(classification_messages(facts, CATEGORIES), "classify", {"product": facts})
            try:
                predicted = json.loads(raw).get("category")
            except json.JSONDecodeError:
                predicted = None
            with session_scope() as s:
                if predicted and predicted != facts["category"]:
                    s.get(Product, product_id).needs_review = True  # type: ignore[union-attr]
                    actions.append(f"classify:mismatch({facts['category']}->{predicted})")
                else:
                    actions.append("classify:ok")
                self._record(s, product_id, "classify", h)
        else:
            actions.append("classify:skipped(idempotent)")
        return actions

    def _fill(self, product_id: int, field: str, text: str, h: str, task: str) -> None:
        with session_scope() as s:
            p = s.get(Product, product_id)
            assert p is not None
            if getattr(p, field):  # someone filled it meanwhile -> never overwrite
                self._record(s, product_id, task, h)
                return
            if self.settings.batch_auto_fill_missing:
                setattr(p, field, text)
                p.content_source, p.needs_review = "ai", True
                p.version += 1
                p.updated_at = datetime.now(UTC)
            self._record(s, product_id, task, h)


# ------------------------------------------------------------------------------------ runner
def run_job(job_id: int, stop_event: threading.Event | None = None) -> str:
    s_cfg = get_settings()
    with session_scope() as s:
        job = s.get(BatchJob, job_id)
        if job is None or job.status in ("running", "completed"):
            return job.status if job else "missing"
        job.status = "running"
        job.started_at = job.started_at or datetime.now(UTC)
        params = dict(job.params)
    BATCH_RUNNING.inc()
    processor = ItemProcessor(job_id, float(params.get("fail_rate") or 0.0))
    t0 = time.perf_counter()
    try:
        while True:
            with session_scope() as s:
                items = list(
                    s.scalars(
                        select(BatchItem)
                        .where(BatchItem.job_id == job_id, BatchItem.status == "pending")
                        .order_by(BatchItem.product_id)
                        .limit(s_cfg.batch_checkpoint_every)
                    )
                )
                work = [(i.id, i.product_id, i.attempts) for i in items]
            if not work:
                break
            if get_store().get_json(f"batch:pause:{job_id}") or (stop_event and stop_event.is_set()):
                _finish(job_id, "paused", processor, t0)
                return "paused"
            results = _run_chunk(processor, work, s_cfg.batch_concurrency, s_cfg.batch_max_attempts)
            _checkpoint(job_id, results)
        status = _final_status(job_id)
        _finish(job_id, status, processor, t0)
        return status
    except Exception as exc:
        log.exception("batch job crashed", extra={"job_id": job_id})
        with session_scope() as s:
            job = s.get(BatchJob, job_id)
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"  # type: ignore[union-attr]
        return "failed"
    finally:
        BATCH_RUNNING.dec()


def _run_chunk(
    processor: ItemProcessor, work: list[tuple[int, int, int]], concurrency: int, max_attempts: int
) -> list[dict[str, Any]]:
    def one(item_id: int, product_id: int, attempts: int) -> dict[str, Any]:
        last_error = ""
        while attempts < max_attempts:
            attempts += 1
            try:
                return {
                    "item_id": item_id,
                    "product_id": product_id,
                    "status": "succeeded",
                    "attempts": attempts,
                    "actions": processor.process(product_id),
                    "error": "",
                }
            except Exception as exc:
                if not is_transient(exc):  # validation / data errors: retrying will not help
                    last_error = f"{type(exc).__name__}: {exc}"
                    break
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(0.05 * 2**attempts, 1.0))  # exponential backoff
        return {
            "item_id": item_id,
            "product_id": product_id,
            "status": "failed",
            "attempts": attempts,
            "actions": [],
            "error": last_error,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return [f.result() for f in as_completed(pool.submit(one, *w) for w in work)]


def _checkpoint(job_id: int, results: list[dict[str, Any]]) -> None:
    from services.search.hybrid import get_search_service

    touched = [r["product_id"] for r in results if r["status"] == "succeeded"]
    with session_scope() as s:
        job = s.get(BatchJob, job_id)
        assert job is not None
        for r in results:
            item = s.get(BatchItem, r["item_id"])
            assert item is not None
            item.status, item.attempts, item.actions = r["status"], r["attempts"], r["actions"]
            item.last_error, item.updated_at = r["error"], datetime.now(UTC)
            skipped = r["status"] == "succeeded" and all(
                "skipped" in a or a.startswith(("validate", "classify:ok")) for a in r["actions"]
            )
            job.processed += 1
            if r["status"] == "failed":
                job.failed += 1
            elif skipped:
                job.skipped += 1
            else:
                job.succeeded += 1
            BATCH_ITEMS.labels(job.job_type, "skipped" if skipped else r["status"]).inc()
        job.checkpoint_product_id = max([r["product_id"] for r in results] + [job.checkpoint_product_id])
        job.stats = {**(job.stats or {}), "heartbeat": datetime.now(UTC).isoformat()}
        # re-embed + re-index the touched products in bulk (search indexes are derived data)
        if touched:
            products = s.scalars(select(Product).where(Product.id.in_(touched))).all()
            stock = available_stock(s, touched)
            codes = supplier_codes(s)
            docs = [product_search_doc(p, stock[p.id], codes.get(p.supplier_id, "")) for p in products]
        else:
            docs = []
    if docs:
        get_search_service().index_docs(docs)
    BATCH_QUEUE.set(_pending_items())


def _pending_items() -> int:
    with session_scope() as s:
        return int(
            s.scalar(select(func.count()).select_from(BatchItem).where(BatchItem.status == "pending")) or 0
        )


def _final_status(job_id: int) -> str:
    with session_scope() as s:
        failed = s.scalar(
            select(func.count())
            .select_from(BatchItem)
            .where(BatchItem.job_id == job_id, BatchItem.status == "failed")
        )
    return "completed_with_errors" if failed else "completed"


def _finish(job_id: int, status: str, processor: ItemProcessor, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    with session_scope() as s:
        job = s.get(BatchJob, job_id)
        assert job is not None
        job.status = status
        if status != "paused":
            job.finished_at = datetime.now(UTC)
        stats = dict(job.stats or {})
        stats["run_seconds"] = round(stats.get("run_seconds", 0) + elapsed, 2)
        stats["tokens"] = stats.get("tokens", 0) + processor.tokens
        stats["items_per_sec"] = round(job.processed / stats["run_seconds"], 2) if stats["run_seconds"] else 0
        stats["success_rate"] = (
            round((job.succeeded + job.skipped) / job.processed, 3) if job.processed else 0
        )
        job.stats = stats
    log.info("batch job finished", extra={"job_id": job_id, "status": status})
