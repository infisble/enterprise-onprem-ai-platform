"""Evaluation harness: run the platform with the configured LLM provider over a fixed dataset and
write per-case CSV + summary JSON to evaluations/reports/.

    python -m evaluations.run_eval                          # current .env provider (mock by default)
    LLM_PROVIDER=vllm VLLM_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ python -m evaluations.run_eval

Running the same dataset against each candidate model (see model_candidates.yaml) gives a like-for-like
comparison of Slovak/Czech quality, extraction, tool selection, RAG, latency and throughput on
NovaPromo's own data - which matters more than public leaderboards for a SK/CS workload.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.core.config import get_settings  # noqa: E402
from packages.core.security import Principal, Role  # noqa: E402
from packages.core.text import detect_language, normalize  # noqa: E402
from packages.observability.tracing import start_trace  # noqa: E402

DATASET = ROOT / "evaluations" / "dataset.jsonl"
REPORTS = ROOT / "evaluations" / "reports"
ECO_MATERIALS = {
    "bamboo",
    "cork",
    "recycled pet",
    "recycled paper",
    "jute",
    "organic cotton",
    "wheat straw",
    "wood",
}


def chrf(hypothesis: str, reference: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (Popović 2015) - robust for morphologically rich SK/CS."""
    hyp, ref = hypothesis.replace(" ", ""), reference.replace(" ", "")
    precisions, recalls = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i : i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i : i + k] for i in range(len(ref) - k + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = statistics.mean(precisions), statistics.mean(recalls)
    return 0.0 if p + r == 0 else (1 + beta**2) * p * r / (beta**2 * p + r)


def _principal(role: str) -> Principal:
    return Principal(f"eval.{role}", Role(role))


def eval_rag(case: dict) -> dict[str, Any]:
    from services.rag.service import get_rag_service

    res = get_rag_service().answer(case["input"], _principal(case["role"]))
    text = normalize(res.answer)
    kw = [k for k in case["expected_keywords"] if normalize(k) in text]
    recall = len(kw) / len(case["expected_keywords"])
    sources = {c.source for c in res.citations} | {c.source for c in res.chunks}
    source_hit = case["expected_source"] in sources
    lang_ok = detect_language(res.answer) == case["expected_language"]
    return {
        "score": round((recall + source_hit) / 2, 3),
        "passed": recall == 1 and source_hit,
        "keyword_recall": round(recall, 3),
        "source_hit": source_hit,
        "language_ok": lang_ok,
        "output": res.answer[:300],
    }


def eval_description(case: dict) -> dict[str, Any]:
    from services.agent.tools import (
        GenerateIn,
        ValidateIn,
        generate_product_description,
        validate_product_data,
    )

    p = _principal("catalog_manager")
    draft = generate_product_description(GenerateIn(sku=case["input"], language=case["language"]), p)
    val = validate_product_data(
        ValidateIn(sku=case["input"], language=case["language"], text=draft["text"]), p
    )
    checks = val["checks"]
    score = sum(checks.values()) / len(checks)
    return {
        "score": round(score, 3),
        "passed": val["valid"] and checks["language_matches"],
        "language_ok": checks["language_matches"],
        "output": draft["text"][:300],
        "errors": "; ".join(val["errors"] + val["warnings"]),
    }


def eval_translation(case: dict) -> dict[str, Any]:
    from services.llm.factory import get_llm
    from services.llm.prompts import translation_messages

    out = get_llm().complete(
        translation_messages(case["input"], case["source"], case["target"]),
        task="translate",
        payload={"text": case["input"], "source": case["source"], "target": case["target"]},
    )
    score = chrf(out.text.strip(), case["reference"])
    return {"score": round(score, 3), "passed": score >= 0.7, "output": out.text.strip()[:300]}


def eval_extraction(case: dict) -> dict[str, Any]:
    from services.search.filters import extract_query

    q = extract_query(case["input"])
    got = {**q.filters.model_dump(), "exact_code": q.exact_code}
    hits = [k for k, v in case["expected"].items() if got.get(k) == v]
    score = len(hits) / len(case["expected"])
    return {
        "score": round(score, 3),
        "passed": score == 1.0,
        "output": json.dumps({k: v for k, v in got.items() if v not in (None, "")}, ensure_ascii=False),
    }


def eval_tool_selection(case: dict) -> dict[str, Any]:
    from services.agent.router import route

    d = route(case["input"], _principal("admin"))
    ok = d.route == case["expected_route"]
    return {"score": float(ok), "passed": ok, "output": f"{d.route} ({d.router}: {d.reason})"}


def _search_check(hit: Any, checks: dict) -> bool:
    ok = True
    if "color" in checks:
        ok &= hit.color == checks["color"]
    if "price_max" in checks:
        ok &= hit.price_eur <= checks["price_max"]
    if "category" in checks:
        ok &= hit.category == checks["category"]
    if "size" in checks:
        ok &= checks["size"] in hit.sizes
    if "sku" in checks:
        ok &= hit.sku == checks["sku"]
    if checks.get("eco"):
        ok &= hit.material in ECO_MATERIALS or "eco" in hit.tags
    if "subcategory_word" in checks:
        ok &= checks["subcategory_word"] in hit.name_en.lower()
    return bool(ok)


def eval_search(case: dict) -> dict[str, Any]:
    from services.search.hybrid import get_search_service

    res = get_search_service().search(case["input"], _principal("sales"), limit=5)
    top = res.results[:5]
    precision = sum(_search_check(h, case["checks"]) for h in top) / max(1, len(top)) if top else 0.0
    return {
        "score": round(precision, 3),
        "passed": precision >= 0.8,
        "output": ", ".join(f"{h.sku} {h.name_en}" for h in top)[:300],
    }


RUNNERS = {
    "sk_qa": eval_rag,
    "cs_qa": eval_rag,
    "rag": eval_rag,
    "product_description": eval_description,
    "translation": eval_translation,
    "extraction": eval_extraction,
    "tool_selection": eval_tool_selection,
    "product_search": eval_search,
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from packages.observability.logging import configure_logging
    from services.search.hybrid import get_search_service

    configure_logging("WARNING", json_logs=False)
    get_search_service().ensure_lexical_loaded()
    s = get_settings()
    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for case in cases:
        with start_trace(username="eval", role="admin", kind="eval", query=case["input"]) as trace:
            t0 = time.perf_counter()
            try:
                result = RUNNERS[case["category"]](case)
            except Exception as exc:
                result = {
                    "score": 0.0,
                    "passed": False,
                    "output": "",
                    "errors": f"{type(exc).__name__}: {exc}",
                }
            latency = (time.perf_counter() - t0) * 1000
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "input": case["input"],
                "score": result["score"],
                "passed": result["passed"],
                "language_ok": result.get("language_ok", ""),
                "latency_ms": round(latency, 1),
                "ttft_ms": trace.ttft_ms or "",
                "input_tokens": trace.input_tokens,
                "output_tokens": trace.output_tokens,
                "output": result.get("output", ""),
                "errors": result.get("errors", ""),
            }
        )

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    llm_ms = sum(r["latency_ms"] for r in rows if r["output_tokens"]) / 1000 or 1
    summary = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": s.llm_provider,
        "model": s.active_model,
        "embeddings": s.embedding_provider,
        "reranker": s.reranker_provider,
        "cases": len(rows),
        "overall": {
            "pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 3),
            "mean_score": round(statistics.mean(r["score"] for r in rows), 3),
            "p50_latency_ms": round(statistics.median(r["latency_ms"] for r in rows), 1),
            "output_tokens_per_sec": round(sum(r["output_tokens"] for r in rows) / llm_ms, 1),
        },
        "categories": {
            cat: {
                "cases": len(rs),
                "pass_rate": round(sum(r["passed"] for r in rs) / len(rs), 3),
                "mean_score": round(statistics.mean(r["score"] for r in rs), 3),
                "language_accuracy": (
                    round(sum(1 for r in rs if r["language_ok"] is True) / len(rs), 3)
                    if any(r["language_ok"] != "" for r in rs)
                    else None
                ),
                "p50_latency_ms": round(statistics.median(r["latency_ms"] for r in rs), 1),
            }
            for cat, rs in sorted(by_cat.items())
        },
        "candidates": yaml.safe_load(
            (ROOT / "evaluations" / "model_candidates.yaml").read_text(encoding="utf-8")
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{s.llm_provider}_{s.active_model.replace('/', '_')}"
    with open(REPORTS / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (REPORTS / f"{stem}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Evaluation: provider={s.llm_provider} model={s.active_model}  cases={len(rows)}")
    print(f"{'category':<22}{'cases':>6}{'pass':>8}{'score':>8}{'lang':>8}{'p50 ms':>9}")
    for cat, m in summary["categories"].items():
        lang = "" if m["language_accuracy"] is None else f"{m['language_accuracy']:.2f}"
        print(
            f"{cat:<22}{m['cases']:>6}{m['pass_rate']:>8.2f}{m['mean_score']:>8.2f}{lang:>8}{m['p50_latency_ms']:>9}"
        )
    o = summary["overall"]
    print(
        f"{'OVERALL':<22}{len(rows):>6}{o['pass_rate']:>8.2f}{o['mean_score']:>8.2f}{'':>8}{o['p50_latency_ms']:>9}"
    )
    print(f"Reports: evaluations/reports/{stem}.json / .csv")


if __name__ == "__main__":
    main()
