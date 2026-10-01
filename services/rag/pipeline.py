"""Document ingestion: files -> parse -> clean -> enrich metadata -> chunk -> scan -> embed -> Qdrant.

Runs offline (bootstrap, nightly job, admin upload), never inside a user request (ADR-010).
Idempotent: a document whose SHA-256 is unchanged is skipped; a changed document has its old
chunks deleted before new ones are written.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from qdrant_client.http import models as qm
from sqlalchemy import select

from packages.core.config import get_settings
from packages.core.db import session_scope
from packages.core.models import Document
from packages.core.security import ACCESS_LEVELS
from packages.core.text import detect_language
from packages.observability.logging import get_logger
from services.llm.embeddings import get_embedder
from services.rag.chunking import chunk_sections
from services.rag.injection import scan
from services.rag.parsers import parse_file
from services.search import vector

log = get_logger(__name__)

DOC_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".json": "json",
    ".xml": "xml",
    ".md": "markdown",
    ".txt": "text",
}


def load_manifest(doc_dir: Path) -> dict[str, dict[str, Any]]:
    path = doc_dir / "manifest.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def ensure_documents_collection(recreate: bool = False) -> None:
    vector.ensure_collection(
        get_settings().documents_collection,
        get_embedder().dim,
        keyword_fields=["access_roles", "filename", "doc_type"],
        numeric_fields=[],
        recreate=recreate,
    )


def ingest_file(path: Path, meta: dict[str, Any], force: bool = False) -> dict[str, Any]:
    s = get_settings()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    access = meta.get("access", "management")  # fail closed: unknown documents are admin-only
    roles = ACCESS_LEVELS[access]
    with session_scope() as db:
        existing = db.scalar(select(Document).where(Document.filename == path.name))
        if existing and existing.sha256 == sha and existing.access_roles == roles and not force:
            return {"file": path.name, "status": "unchanged", "chunks": existing.chunk_count}

    sections = parse_file(path, meta)
    chunks = chunk_sections(sections)
    title = meta.get("title", path.stem)
    doc_type = DOC_TYPES[path.suffix.lower()]
    embedder = get_embedder()
    vectors = embedder.embed_documents([f"{title}. {c.text}" for c in chunks])
    flagged = 0
    points = []
    for c, vec in zip(chunks, vectors, strict=True):
        inj = scan(c.text)
        flagged += inj.flagged
        points.append(
            (
                vector.chunk_point_id(f"{path.name}#{c.index}"),
                vec,
                {
                    "filename": path.name,
                    "title": title,
                    "doc_type": doc_type,
                    "text": c.text,
                    "heading": c.heading,
                    "locator": c.locator,
                    "chunk_index": c.index,
                    "kind": c.kind,
                    "access_level": access,
                    "access_roles": roles,
                    "classification": meta.get("classification", "internal"),
                    "language": detect_language(c.text),
                    "injection_flagged": inj.flagged,
                    "injection_matches": inj.matches,
                    "sha256": sha,
                },
            )
        )

    client = vector.get_qdrant()
    client.delete(
        s.documents_collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(must=[qm.FieldCondition(key="filename", match=qm.MatchValue(value=path.name))])
        ),
    )
    vector.upsert(s.documents_collection, points)

    with session_scope() as db:
        doc = db.scalar(select(Document).where(Document.filename == path.name))
        if doc is None:
            doc = Document(filename=path.name)
            db.add(doc)
        doc.doc_type, doc.title, doc.access_roles = doc_type, title, roles
        doc.classification, doc.sha256 = meta.get("classification", "internal"), sha
        doc.chunk_count, doc.flagged_chunks = len(chunks), flagged
    if flagged:
        log.warning(
            "prompt-injection patterns found during ingestion", extra={"file": path.name, "chunks": flagged}
        )
    return {
        "file": path.name,
        "status": "ingested",
        "chunks": len(chunks),
        "flagged_chunks": flagged,
        "access": access,
    }


def ingest_directory(doc_dir: Path, force: bool = False) -> list[dict[str, Any]]:
    ensure_documents_collection()
    manifest = load_manifest(doc_dir)
    results = []
    for path in sorted(doc_dir.iterdir()):
        if path.name == "manifest.yaml" or path.suffix.lower() not in DOC_TYPES:
            continue
        results.append(ingest_file(path, manifest.get(path.name, {}), force=force))
    return results
