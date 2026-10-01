"""Deterministic validation of generated product content against the copy guidelines.

LLM output is never trusted blindly: these checks run before any proposal is created and their
results are stored with the approval request so the approver sees them.
"""

from __future__ import annotations

import re
from typing import Any

from packages.core.text import MATERIAL_NAMES, detect_language, normalize
from services.rag.injection import scan

FORBIDDEN_WORDS = [
    "najlepší",
    "najlepšia",
    "nejlepší",
    "best",
    "revolutionary",
    "revolučný",
    "unique",
    "jedinečný",
    "jedinečná",
    "guaranteed",
    "garantovaný",
]
COMPETITOR_OR_SUPPLIER = ["eurogift", "promotex", "ecoline"]


def validate_description(text: str, language: str, product: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    n = len(text)
    checks["length_350_600"] = 350 <= n <= 600
    if not checks["length_350_600"]:
        (errors if n < 200 or n > 800 else warnings).append(f"length {n} chars (guideline 350-600)")
    checks["no_exclamation"] = "!" not in text
    if not checks["no_exclamation"]:
        errors.append("contains exclamation marks")
    lowered = text.lower()
    bad = [w for w in FORBIDDEN_WORDS if re.search(rf"\b{re.escape(w)}\b", lowered)]
    checks["no_superlatives"] = not bad
    if bad:
        errors.append(f"forbidden superlatives: {', '.join(bad)}")
    checks["no_prices_or_stock"] = not re.search(
        r"\d+[.,]?\d*\s*(€|eur\b)|skladom|na sklade|in stock", lowered
    )
    if not checks["no_prices_or_stock"]:
        errors.append("mentions prices or stock (these change daily)")
    names = [s for s in COMPETITOR_OR_SUPPLIER if s in normalize(text)]
    checks["no_supplier_names"] = not names
    if names:
        errors.append(f"mentions supplier names: {', '.join(names)}")
    material = product.get("material", "")
    variants = [material, *MATERIAL_NAMES.get(material, ())]
    checks["mentions_material"] = any(normalize(v) in normalize(text) for v in variants if v)
    if not checks["mentions_material"]:
        warnings.append("material is not mentioned")
    detected = detect_language(text)
    checks["language_matches"] = detected == language or (language == "cs" and detected == "sk")
    if detected != language:
        warnings.append(f"detected language '{detected}', expected '{language}'")
    inj = scan(text)
    checks["no_injected_instructions"] = not inj.flagged
    if inj.flagged:
        errors.append("generated text contains instruction-like content")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "checks": checks, "length": n}
