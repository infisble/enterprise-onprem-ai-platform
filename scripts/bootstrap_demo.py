"""Bootstrap the demo: schema -> users -> supplier feeds -> SQL -> search indexes -> documents.

Idempotent: safe to run repeatedly. `--reset` drops SQL tables and recreates indexes.

    python scripts/bootstrap_demo.py            # local (SQLite + embedded Qdrant + in-process BM25)
    docker compose exec api python scripts/bootstrap_demo.py   # Docker (Postgres + Qdrant + Elasticsearch)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.core.config import get_settings  # noqa: E402
from packages.core.db import Base, get_engine, init_db, session_scope  # noqa: E402
from packages.observability.logging import configure_logging  # noqa: E402
from services.ingestion.loader import seed_customers_and_orders, seed_users, upsert_products  # noqa: E402
from services.ingestion.suppliers import load_all_feeds  # noqa: E402
from services.rag.pipeline import ensure_documents_collection, ingest_directory  # noqa: E402
from services.search.hybrid import get_search_service  # noqa: E402


def wait_for_dependencies(timeout_s: int = 120) -> None:
    s = get_settings()
    deadline = time.time() + timeout_s
    checks = []
    if s.elasticsearch_url:
        from services.search.lexical import get_lexical_index

        checks.append(("elasticsearch", lambda: get_lexical_index().ping()))
    if s.qdrant_url:
        from services.search.vector import get_qdrant

        checks.append(("qdrant", lambda: get_qdrant().get_collections() is not None))
    for name, check in checks:
        while True:
            try:
                if check():
                    print(f"  ✓ {name} reachable")
                    break
            except Exception:
                pass
            if time.time() > deadline:
                raise SystemExit(f"{name} not reachable after {timeout_s}s")
            time.sleep(2)


def load_sql(is_sqlite: bool, data_dir: Path) -> None:
    if not is_sqlite:
        from services.agent.analytics_sql import apply_readonly_grants

        print(f"  ✓ analytics read-only grants: {apply_readonly_grants()}")
    with session_scope() as db:
        print(f"  ✓ demo users created: {seed_users(db)}")

    report = load_all_feeds(data_dir / "suppliers")
    with session_scope() as db:
        stats = upsert_products(db, report, data_dir)
    print(f"  ✓ supplier feeds: {json.dumps(report.per_supplier)}")
    print(f"    products: {stats}")
    for r in report.rejected:
        print(f"    ✗ rejected {r['source']}: {r['reason']}")

    with session_scope() as db:
        n = seed_customers_and_orders(db)
    print(f"  ✓ customers/orders seeded: {n} customers")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="drop and recreate everything")
    parser.add_argument(
        "--if-empty", action="store_true", help="skip when products already exist (container start)"
    )
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    configure_logging("WARNING", json_logs=False)
    s = get_settings()
    data_dir = Path(s.data_dir)
    t0 = time.time()
    print(
        f"NovaPromo bootstrap  (db={s.database_url.split('@')[-1]}, qdrant={s.qdrant_url or 'embedded'}, "
        f"lexical={s.elasticsearch_url or 'in-process BM25'}, llm={s.llm_provider})"
    )
    wait_for_dependencies()

    if args.reset:
        Base.metadata.drop_all(get_engine())
    init_db()
    indexes_only = False
    if args.if_empty:
        from sqlalchemy import func, select

        from packages.core.models import Product
        from services.search import vector
        from services.search.lexical import get_lexical_index

        with session_scope() as db:
            has_products = bool(db.scalar(select(func.count()).select_from(Product)))
        if has_products:
            get_search_service().ensure_lexical_loaded()
            indexes_ok = (
                vector.count(s.products_collection) > 0
                and vector.count(s.documents_collection) > 0
                and get_lexical_index().count() > 0
            )
            if indexes_ok:
                print("  ✓ data and indexes present - skipping bootstrap (use --reset to rebuild)")
                return
            # SQL is the source of truth; search indexes are derived and can always be rebuilt.
            print("  ! SQL data present but search indexes missing - rebuilding indexes from SQL")
            indexes_only = True
    if not indexes_only:
        load_sql(s.is_sqlite, data_dir)

    idx = get_search_service().index_all(recreate=True)
    print(f"  ✓ product indexes: {idx}")

    ensure_documents_collection(recreate=args.reset)
    for r in ingest_directory(data_dir / "documents", force=args.reset or indexes_only):
        flag = (
            f", ⚠ {r['flagged_chunks']} chunk(s) flagged for prompt injection"
            if r.get("flagged_chunks")
            else ""
        )
        print(
            f"  ✓ {r['file']}: {r['status']} ({r['chunks']} chunks{', access=' + r['access'] if 'access' in r else ''}{flag})"
        )
    from services.search.vector import reset_qdrant

    reset_qdrant()
    print(
        f"Done in {time.time() - t0:.1f}s. Demo users (password '{s.demo_password}'): eva.employee, samuel.sales, "
        "katarina.catalog, adam.admin"
    )


if __name__ == "__main__":
    main()
