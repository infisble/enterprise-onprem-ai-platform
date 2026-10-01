from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select

from apps.api.deps import get_principal, require
from packages.core.db import session_scope
from packages.core.models import BatchItem, BatchJob, ProductUpdateRequest
from packages.core.security import AuthorizationError, Permission, Principal
from packages.schemas.workflow import BatchJobIn, BatchJobOut, DecisionIn, UpdateRequestOut
from services.agent import approvals
from services.batch.jobs import create_job, request_pause, resume_job

router = APIRouter(prefix="/api", tags=["workflow"])


def _req_out(r: ProductUpdateRequest) -> UpdateRequestOut:
    return UpdateRequestOut.model_validate(r, from_attributes=True)


@router.get("/approvals", response_model=list[UpdateRequestOut])
def list_approvals(status: str | None = Query(None), principal: Principal = Depends(get_principal)):
    if not (principal.can(Permission.UPDATES_APPROVE) or principal.can(Permission.UPDATES_REQUEST)):
        raise HTTPException(403, "Not allowed to view change requests")
    with session_scope() as s:
        q = select(ProductUpdateRequest).order_by(ProductUpdateRequest.id.desc()).limit(200)
        if status:
            q = q.where(ProductUpdateRequest.status == status)
        if not principal.can(Permission.UPDATES_APPROVE):  # requesters see their own requests
            q = q.where(ProductUpdateRequest.requested_by == principal.username)
        return [_req_out(r) for r in s.scalars(q)]


@router.post("/approvals/{request_id}/approve", response_model=UpdateRequestOut)
def approve(
    request_id: int, body: DecisionIn, principal: Principal = Depends(require(Permission.UPDATES_APPROVE))
):
    """Approve and publish in one action (publish re-verifies payload hash and product version)."""
    try:
        approvals.approve(request_id, principal, body.note)
        req = approvals.publish(request_id, principal)
    except (approvals.ApprovalError, AuthorizationError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return _req_out(req)


@router.post("/approvals/{request_id}/reject", response_model=UpdateRequestOut)
def reject(
    request_id: int, body: DecisionIn, principal: Principal = Depends(require(Permission.UPDATES_APPROVE))
):
    try:
        return _req_out(approvals.reject(request_id, principal, body.note))
    except approvals.ApprovalError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/batch/jobs", response_model=list[BatchJobOut])
def list_jobs(principal: Principal = Depends(require(Permission.BATCH_READ))):
    with session_scope() as s:
        return [
            BatchJobOut.model_validate(j, from_attributes=True)
            for j in s.scalars(select(BatchJob).order_by(BatchJob.id.desc()).limit(50))
        ]


@router.post("/batch/jobs", response_model=BatchJobOut)
def start_job(body: BatchJobIn, principal: Principal = Depends(require(Permission.BATCH_RUN))):
    job_id = create_job(
        body.job_type,
        created_by=principal.username,
        limit=body.limit,
        only_missing=body.only_missing,
        fail_rate=body.fail_rate,
    )
    with session_scope() as s:
        return BatchJobOut.model_validate(s.get(BatchJob, job_id), from_attributes=True)


def _item_out(i: BatchItem) -> dict:
    return {
        "product_id": i.product_id,
        "status": i.status,
        "attempts": i.attempts,
        "actions": i.actions,
        "error": i.last_error,
    }


@router.get("/batch/jobs/{job_id}")
def job_detail(job_id: int, principal: Principal = Depends(require(Permission.BATCH_READ))) -> dict:
    with session_scope() as s:
        job = s.get(BatchJob, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        counts = dict(
            s.execute(
                select(BatchItem.status, func.count())
                .where(BatchItem.job_id == job_id)
                .group_by(BatchItem.status)
            ).all()
        )
        recent = s.scalars(
            select(BatchItem)
            .where(BatchItem.job_id == job_id, BatchItem.status != "pending")
            .order_by(BatchItem.updated_at.desc())
            .limit(25)
        ).all()
        failed = s.scalars(
            select(BatchItem).where(BatchItem.job_id == job_id, BatchItem.status == "failed").limit(25)
        ).all()
        return {
            "job": BatchJobOut.model_validate(job, from_attributes=True).model_dump(),
            "item_counts": counts,
            "recent_items": [_item_out(i) for i in recent],
            "failed_items": [_item_out(i) for i in failed],
        }


@router.post("/batch/jobs/{job_id}/pause")
def pause(job_id: int, principal: Principal = Depends(require(Permission.BATCH_RUN))) -> dict:
    request_pause(job_id)
    return {"job_id": job_id, "pause_requested": True}


@router.post("/batch/jobs/{job_id}/resume")
def resume(job_id: int, principal: Principal = Depends(require(Permission.BATCH_RUN))) -> dict:
    try:
        resume_job(job_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job_id": job_id, "status": "queued"}
