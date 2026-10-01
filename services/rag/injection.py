"""Prompt-injection detection for untrusted retrieved content.

Detection is a *signal*, not the security boundary. Even an undetected injection cannot cause harm
because: (1) retrieved text is wrapped as untrusted data in the prompt, (2) the agent's tool plan is
decided before generation and never from document content, (3) there is no tool that can export
customer data, (4) every tool call is RBAC-checked and writes need human approval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override_instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.]{0,40}\b(previous|prior|above|all|earlier|system)\b[^.]{0,20}"
            r"\b(instructions?|prompts?|rules?)\b",
            re.I,
        ),
    ),
    ("role_hijack", re.compile(r"\b(you are now|act as|pretend to be|new instructions)\b", re.I)),
    (
        "addressed_to_ai",
        re.compile(r"\b(for|to) (ai|llm|the) (assistants?|models?|agents?)\b|\bAI ASSISTANTS?\b", re.I),
    ),
    (
        "exfiltration",
        re.compile(
            r"\b(export|dump|send|reveal|disclose|list|print)\b[^.]{0,40}\b(all )?(customer|client|user|password|"
            r"credential|e-?mail|system prompt|records?)\b",
            re.I,
        ),
    ),
    (
        "fake_authority",
        re.compile(
            r"\b(authori[sz]ed by|on behalf of|approved by) (\w+ ){0,2}(management|ceo|it department|admin|director)",
            re.I,
        ),
    ),
    (
        "sk_cs_override",
        re.compile(
            r"\b(ignoruj|ignorujte|zabudni|zapomeň)\b[^.]{0,40}\b(pokyny|inštrukcie|instrukce)\b", re.I
        ),
    ),
]

REDACTION = "[removed: suspected prompt-injection instruction]"


@dataclass
class InjectionScan:
    flagged: bool
    matches: list[str] = field(default_factory=list)
    sanitized: str = ""


STRONG = {"override_instructions", "role_hijack", "sk_cs_override", "exfiltration"}


def scan(text: str) -> InjectionScan:
    # PDF extraction breaks lines mid-phrase ("system<newline>prompt"); scan on normalised whitespace.
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    hits_per_sentence = [[name for name, pattern in PATTERNS if pattern.search(s)] for s in sentences]
    # A chunk is flagged by a strong signal or by two weak signals in one sentence. Once flagged, any
    # sentence with even a weak signal (e.g. "authorised by management") is removed as well.
    flagged = any(STRONG & set(h) or len(h) >= 2 for h in hits_per_sentence)
    matches: list[str] = []
    kept: list[str] = []
    for sent, hits in zip(sentences, hits_per_sentence, strict=True):
        if flagged and hits:
            matches.extend(hits)
            if not kept or kept[-1] != REDACTION:
                kept.append(REDACTION)
        else:
            kept.append(sent)
    return InjectionScan(flagged=flagged, matches=sorted(set(matches)), sanitized=" ".join(kept))


def scan_user_message(text: str) -> InjectionScan:
    """User messages are trusted for intent but still screened for obvious jailbreak attempts."""
    return scan(text)
