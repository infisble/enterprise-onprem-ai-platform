"""Batch worker + nightly scheduler.

    python -m services.batch.worker            # dedicated worker process (Docker service `worker`)

In `BATCH_MODE=inline` (laptop demo) the API starts the same worker loop in a background thread.
Scale-out: run N worker containers; each pops job ids from the Redis list (BRPOP is atomic), and
per-item rows + the idempotency ledger make concurrent/resumed processing safe.
"""

from __future__ import annotations

import signal
import threading
import time
from datetime import datetime

from packages.core.cache import get_store
from packages.core.config import get_settings
from packages.core.db import init_db
from packages.observability.logging import configure_logging, get_logger
from services.batch.jobs import QUEUE, create_job, run_job

log = get_logger(__name__)


class Worker:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def loop(self) -> None:
        store = get_store()
        log.info("batch worker started", extra={"backend": store.backend})
        while not self.stop_event.is_set():
            msg = store.pop(QUEUE, timeout_s=2)
            if msg:
                status = run_job(int(msg["job_id"]), self.stop_event)
                log.info("job processed", extra={"job_id": msg["job_id"], "status": status})

    def start_background(self) -> None:
        self._thread = threading.Thread(target=self.loop, name="batch-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()


class NightlyScheduler:
    """Creates one `nightly_enrichment` job per day at BATCH_SCHEDULE_HOUR (local time).
    A Redis key guarantees exactly one job per day even with several worker replicas."""

    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event = stop_event

    def loop(self) -> None:
        hour = get_settings().batch_schedule_hour
        while not self.stop_event.is_set():
            now = datetime.now()
            key = f"batch:nightly:{now.date().isoformat()}"
            store = get_store()
            # SET NX: exactly one replica wins the right to create tonight's job
            if now.hour == hour and store.set_if_absent(key, True, ttl_s=26 * 3600):
                create_job("nightly_enrichment", created_by="scheduler")
            self.stop_event.wait(60)


def main() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    init_db()
    from services.search.hybrid import get_search_service

    get_search_service().ensure_lexical_loaded()
    from services.batch.jobs import recover_stale_jobs

    recovered = recover_stale_jobs()
    if recovered:
        log.warning("re-queued stale jobs after restart", extra={"jobs": recovered})
    worker = Worker()
    signal.signal(signal.SIGTERM, lambda *_: worker.stop())
    signal.signal(signal.SIGINT, lambda *_: worker.stop())
    threading.Thread(target=NightlyScheduler(worker.stop_event).loop, name="scheduler", daemon=True).start()
    worker.loop()
    time.sleep(0.1)


if __name__ == "__main__":
    main()
