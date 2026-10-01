"""Human-in-the-loop write boundary for product data.

propose (agent / batch / human)  ->  pending
approve (admin, not the requester) ->  approved   (separation of duties)
publish                              ->  published (payload hash re-verified, optimistic version check)
                                     ->  conflict  (product changed since the proposal was made)
reject                               ->  rejected
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy import select

from packages.core.db import session_scope
from packages.core.models import AuditLog, Product, ProductUpdateRequest
from packages.core.security import AuthorizationError, Permission, Principal
from packages.observability.tracing import current_trace

EDITABLE_FIELDS = {"description_sk", "description_cs", "description_en", "name_sk", "name_cs"}


class ApprovalError(Exception):
    pass


def payload_hash(sku: str, field: str, new_value: str, base_version: int) -> str:
    return hashlib.sha256(f"{sku}|{field}|{base_version}|{new_value}".encode()).hexdigest()


def _audit(
    session, principal: Principal, action: str, req: ProductUpdateRequest, outcome: str, **details
) -> None:
    trace = current_trace()
    session.add(
        AuditLog(
            request_id=trace.request_id if trace else req.request_id,
            username=principal.username,
            role=principal.role.value,
            action=action,
            resource=f"update_request:{req.id} {req.sku}.{req.field}",
            outcome=outcome,
            details={"payload_hash": req.payload_hash, **details},
        )
    )


def create_request(
    principal: Principal,
    sku: str,
    field: str,
    new_value: str,
    rationale: str,
    validation: dict,
    source: str = "agent",
) -> ProductUpdateRequest:
    if not principal.can(Permission.UPDATES_REQUEST):
        raise AuthorizationError(principal, Permission.UPDATES_REQUEST)
    if field not in EDITABLE_FIELDS:
        raise ApprovalError(f"field '{field}' is not editable through the assistant")
    with session_scope() as s:
        product = s.scalar(select(Product).where(Product.sku == sku))
        if product is None:
            raise ApprovalError(f"unknown product {sku}")
        h = payload_hash(sku, field, new_value, product.version)
        dup = s.scalar(
            select(ProductUpdateRequest).where(
                ProductUpdateRequest.payload_hash == h, ProductUpdateRequest.status == "pending"
            )
        )
        if dup is not None:  # idempotent: same proposal twice -> same request
            return dup
        trace = current_trace()
        req = ProductUpdateRequest(
            product_id=product.id,
            sku=sku,
            field=field,
            old_value=getattr(product, field) or "",
            new_value=new_value,
            payload_hash=h,
            base_version=product.version,
            status="pending",
            source=source,
            validation=validation,
            rationale=rationale,
            request_id=trace.request_id if trace else "",
            requested_by=principal.username,
        )
        s.add(req)
        s.flush()
        _audit(s, principal, "update_request:create", req, "success", field=field)
        return req


def approve(request_id: int, principal: Principal, note: str = "") -> ProductUpdateRequest:
    if not principal.can(Permission.UPDATES_APPROVE):
        raise AuthorizationError(principal, Permission.UPDATES_APPROVE, f"update request {request_id}")
    with session_scope() as s:
        req = s.get(ProductUpdateRequest, request_id)
        if req is None:
            raise ApprovalError("request not found")
        if req.status != "pending":
            raise ApprovalError(f"request is {req.status}, not pending")
        if req.requested_by == principal.username:
            _audit(s, principal, "update_request:approve", req, "denied", reason="self-approval")
            raise ApprovalError("separation of duties: you cannot approve your own request")
        req.status, req.decided_by, req.decided_at, req.decision_note = (
            "approved",
            principal.username,
            datetime.now(UTC),
            note,
        )
        _audit(s, principal, "update_request:approve", req, "success", note=note)
        return req


def reject(request_id: int, principal: Principal, note: str = "") -> ProductUpdateRequest:
    if not principal.can(Permission.UPDATES_APPROVE):
        raise AuthorizationError(principal, Permission.UPDATES_APPROVE, f"update request {request_id}")
    with session_scope() as s:
        req = s.get(ProductUpdateRequest, request_id)
        if req is None or req.status != "pending":
            raise ApprovalError("request not found or not pending")
        req.status, req.decided_by, req.decided_at, req.decision_note = (
            "rejected",
            principal.username,
            datetime.now(UTC),
            note,
        )
        _audit(s, principal, "update_request:reject", req, "success", note=note)
        return req


def publish(request_id: int, principal: Principal) -> ProductUpdateRequest:
    if not principal.can(Permission.UPDATES_APPROVE):
        raise AuthorizationError(principal, Permission.UPDATES_APPROVE, f"update request {request_id}")
    with session_scope() as s:
        req = s.get(ProductUpdateRequest, request_id)
        if req is None or req.status != "approved":
            raise ApprovalError("only approved requests can be published")
        product = s.get(Product, req.product_id)
        assert product is not None
        if payload_hash(req.sku, req.field, req.new_value, req.base_version) != req.payload_hash:
            _audit(s, principal, "update_request:publish", req, "denied", reason="payload hash mismatch")
            raise ApprovalError("payload hash mismatch - proposal was modified after approval")
        if product.version != req.base_version:
            req.status = "conflict"
            _audit(
                s,
                principal,
                "update_request:publish",
                req,
                "error",
                reason="version conflict",
                base_version=req.base_version,
                current_version=product.version,
            )
            return req
        setattr(product, req.field, req.new_value)
        product.version += 1
        product.content_source = "ai" if req.source in ("agent", "batch") else "human"
        product.needs_review = False
        product.updated_at = datetime.now(UTC)
        req.status, req.published_at = "published", datetime.now(UTC)
        _audit(s, principal, "update_request:publish", req, "success", new_version=product.version)
        pid = product.id
    _reindex(pid)
    return req


def _reindex(product_id: int) -> None:
    from packages.core.repository import available_stock, supplier_codes
    from services.search.documents import product_search_doc
    from services.search.hybrid import get_search_service

    with session_scope() as s:
        p = s.get(Product, product_id)
        if p is None:
            return
        doc = product_search_doc(
            p, available_stock(s, [p.id])[p.id], supplier_codes(s).get(p.supplier_id, "")
        )
    get_search_service().index_docs([doc])
