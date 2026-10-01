from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel


def _as_utc(value: datetime) -> datetime:
    """All timestamps are stored in UTC; SQLite returns them naive, so re-attach the zone."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


UTCDateTime = Annotated[datetime, AfterValidator(_as_utc)]


class UpdateRequestOut(BaseModel):
    id: int
    sku: str
    field: str
    old_value: str
    new_value: str
    payload_hash: str
    base_version: int
    status: str
    source: str
    validation: dict[str, Any]
    rationale: str
    request_id: str
    requested_by: str
    requested_at: UTCDateTime
    decided_by: str | None
    decided_at: UTCDateTime | None
    decision_note: str
    published_at: UTCDateTime | None


class DecisionIn(BaseModel):
    note: str = ""


class BatchJobIn(BaseModel):
    job_type: str = "nightly_enrichment"
    limit: int | None = None
    only_missing: bool = True
    fail_rate: float = 0.0  # fault injection for demonstrating retries (0..0.5)


class BatchJobOut(BaseModel):
    id: int
    job_type: str
    status: str
    params: dict[str, Any]
    created_by: str
    created_at: UTCDateTime
    started_at: UTCDateTime | None
    finished_at: UTCDateTime | None
    total: int
    processed: int
    succeeded: int
    failed: int
    skipped: int
    checkpoint_product_id: int
    stats: dict[str, Any]
    error: str
