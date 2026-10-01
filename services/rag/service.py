"""Permission-aware RAG with citations.

    question ─► embed ─► Qdrant ANN *with access_roles filter* (RBAC inside the index)
             ─► defence-in-depth re-check of every chunk's roles (fail closed)
             ─► rerank (vector score + lexical overlap + heading match)
             ─► injection scan + sanitisation ─► context budget
             ─► LLM (documents wrapped as untrusted data) ─► answer + citations

Forbidden chunks are removed BEFORE the context is assembled, so the model never sees them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client.http import models as qm

from packages.core.config import get_settings
from packages.core.security import Principal
from packages.core.text import detect_language, stem, tokenize
from packages.observability.metrics import SECURITY_EVENTS
from packages.observability.tracing import span
from packages.schemas.chat import Citation
from services.llm.embeddings import get_embedder
from services.llm.factory import get_llm
from services.llm.prompts import rag_messages
from services.rag.injection import scan
from services.search import vector


@dataclass
class RetrievedChunk:
    ref: int
    text: str
    source: str
    title: str
    doc_type: str
    locator: dict[str, Any]
    score: float
    access_roles: list[str]
    flagged: bool = False
    injection_matches: list[str] = field(default_factory=list)


@dataclass
class RAGResult:
    answer: str
    citations: list[Citation]
    chunks: list[RetrievedChunk]
    filtered_by_rbac: int
    injection_flags: int
    language: str


def _role_filter(principal: Principal) -> qm.Filter:
    return qm.Filter(
        must=[qm.FieldCondition(key="access_roles", match=qm.MatchAny(any=[principal.role.value]))]
    )


class RAGService:
    def __init__(self, top_k: int = 12, context_chunks: int = 5, context_char_budget: int = 6000) -> None:
        self.top_k, self.context_chunks, self.budget = top_k, context_chunks, context_char_budget

    def retrieve(self, question: str, principal: Principal) -> tuple[list[RetrievedChunk], int]:
        collection = get_settings().documents_collection
        with span("rag_retrieval", kind="retrieval", role=principal.role.value) as a:
            qvec = get_embedder().embed_query(question)
            hits = vector.search(collection, qvec, _role_filter(principal), self.top_k)
            # Observability only: how many higher-ranked chunks the role could NOT see. Their text is
            # never read or returned; only ids and roles are compared.
            unfiltered = vector.search(collection, qvec, None, self.top_k)
            allowed_ids = {h[0] for h in hits}
            filtered_out = sum(
                1
                for pid, _, payload in unfiltered
                if pid not in allowed_ids and principal.role.value not in payload.get("access_roles", [])
            )
            a.update(hits=len(hits), filtered_by_rbac=filtered_out)
        if filtered_out:
            SECURITY_EVENTS.labels(event="rag_chunks_filtered").inc(filtered_out)

        q_terms = {stem(t) for t in tokenize(question)}
        candidates: list[tuple[float, RetrievedChunk]] = []
        with span("rag_rerank", kind="rerank", candidates=len(hits)):
            for _, score, p in hits:
                if principal.role.value not in p.get("access_roles", []):  # defence in depth, fail closed
                    SECURITY_EVENTS.labels(event="rag_post_filter_block").inc()
                    continue
                c_terms = {stem(t) for t in tokenize(p["text"])}
                h_terms = {stem(t) for t in tokenize(p.get("heading", ""))}
                overlap = len(q_terms & c_terms) / max(1, len(q_terms))
                heading_bonus = 0.15 if q_terms & h_terms else 0.0
                final = 0.55 * score + 0.35 * overlap + heading_bonus
                candidates.append(
                    (
                        final,
                        RetrievedChunk(
                            ref=0,
                            text=p["text"],
                            source=p["filename"],
                            title=p.get("title", ""),
                            doc_type=p["doc_type"],
                            locator=p.get("locator", {}),
                            score=round(final, 4),
                            access_roles=p.get("access_roles", []),
                            flagged=bool(p.get("injection_flagged")),
                            injection_matches=p.get("injection_matches", []),
                        ),
                    )
                )
        candidates.sort(key=lambda x: -x[0])
        chunks: list[RetrievedChunk] = []
        used = 0
        for _, c in candidates:
            if len(chunks) >= self.context_chunks or used + len(c.text) > self.budget:
                break
            c.ref = len(chunks) + 1
            chunks.append(c)
            used += len(c.text)
        return chunks, filtered_out

    def answer(self, question: str, principal: Principal) -> RAGResult:
        chunks, filtered = self.retrieve(question, principal)
        lang = detect_language(question)
        injection_flags = 0
        context: list[dict[str, Any]] = []
        for c in chunks:
            result = scan(c.text)  # re-scan at query time: ingestion-time rules may be outdated
            flagged = c.flagged or result.flagged
            if flagged:
                injection_flags += 1
                SECURITY_EVENTS.labels(event="prompt_injection_detected").inc()
                c.flagged = True
                c.text = result.sanitized
            context.append({"ref": c.ref, "source": c.source, "text": c.text, "flagged": flagged})
        if not context:
            answer = {
                "sk": "Nenašiel som žiadne dokumenty, ku ktorým máte prístup a ktoré by odpovedali na otázku.",
                "cs": "Nenašel jsem žádné dokumenty, ke kterým máte přístup a které by odpovídaly na otázku.",
            }.get(lang, "I found no documents you are allowed to access that answer this question.")
            return RAGResult(answer, [], [], filtered, 0, lang)
        llm_result = get_llm().complete(
            rag_messages(question, context),
            task="rag_answer",
            payload={"question": question, "chunks": context, "language": lang},
        )
        citations = self.citations(llm_result.text, chunks)
        return RAGResult(llm_result.text, citations, chunks, filtered, injection_flags, lang)

    @staticmethod
    def citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
        import re

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        selected = [c for c in chunks if c.ref in cited] or chunks[:3]
        return [
            Citation(
                ref=c.ref,
                source=c.source,
                doc_type=c.doc_type,
                locator=c.locator,
                snippet=(c.text[:240] + "…") if len(c.text) > 240 else c.text,
                score=c.score,
            )
            for c in selected
        ]


_rag: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag
    if _rag is None:
        _rag = RAGService()
    return _rag
