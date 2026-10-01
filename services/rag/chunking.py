"""Cleaning + structure-aware chunking.

Prose is split on sentence boundaries *within* a section (never across headings), targeting ~800
characters with one sentence of overlap; the heading is prepended so each chunk is self-describing.
Records (spreadsheet rows, JSON/XML items) are never split or merged: one record = one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from services.rag.parsers import Section

BOILERPLATE = [
    re.compile(p, re.I)
    for p in (
        r"NovaPromo s\.r\.o\. - internal document - synthetic demo data",
        r"^Page \d+ of \d+$",
    )
]


def clean(text: str) -> str:
    for pattern in BOILERPLATE:
        text = pattern.sub(" ", text)
    text = text.replace("­", "")  # soft hyphens from PDF extraction
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text) if "- " in text else text
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


@dataclass
class Chunk:
    text: str
    locator: dict[str, Any]
    heading: str
    kind: str
    index: int


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-ZÁČĎÉÍĽĹŇÓŔŠŤÚÝŽ0-9])", text) if s.strip()]


def chunk_sections(sections: list[Section], max_chars: int = 800, overlap_sentences: int = 1) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in sections:
        body = clean(sec.text)
        if not body:
            continue
        prefix = f"{sec.heading}\n" if sec.heading and sec.kind == "prose" else ""
        if sec.kind == "record" or len(body) <= max_chars:
            chunks.append(Chunk(prefix + body, sec.locator, sec.heading, sec.kind, len(chunks)))
            continue
        sentences = _split_sentences(body)
        buf: list[str] = []
        for sent in sentences:
            if buf and sum(len(s) + 1 for s in buf) + len(sent) > max_chars:
                chunks.append(Chunk(prefix + " ".join(buf), sec.locator, sec.heading, sec.kind, len(chunks)))
                buf = buf[-overlap_sentences:] if overlap_sentences else []
            buf.append(sent)
        if buf:
            chunks.append(Chunk(prefix + " ".join(buf), sec.locator, sec.heading, sec.kind, len(chunks)))
    return chunks
