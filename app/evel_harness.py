"""
SHL Assessment Recommender - Evaluation Harness
=================================================

Parses the provided GenAI_SampleConversations/*.md transcripts, replays each
conversation turn-by-turn against the live /chat endpoint, and computes
REAL metrics from what the deployed system actually returns.

No metric is reported unless it is actually computed from observed data.
Recall@10 / precision-style retrieval metrics are DELIBERATELY OMITTED
because the traces do not include a ground-truth relevance judgement set
(i.e. we don't know the "correct" full list of relevant SHL assessments
for each query), so any Recall@10 number would be fabricated.

Usage:
    python -m app.eval_harness --base-url https://your-app.onrender.com --traces-dir app/GenAI_SampleConversations

Output:
    - Per-trace pass/fail breakdown printed to console
    - eval_results.json with raw per-trace data
    - eval_report.md with a markdown table ready to paste into your report
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests


# ---------------------------------------------------------------------------
# 1. Parse .md transcripts into a list of turns: [{role, content}, ...]
# ---------------------------------------------------------------------------

TURN_HEADER_RE = re.compile(r"^###\s*Turn\s*\d+", re.MULTILINE)
USER_BLOCK_RE = re.compile(r"\*\*User\*\*\s*\n+>\s?(.*?)(?=\n\n\*\*Agent\*\*)", re.DOTALL)
AGENT_BLOCK_RE = re.compile(r"\*\*Agent\*\*\s*\n+(.*?)(?=\n_`end_of_conversation`|\n_No recommendations|\Z)", re.DOTALL)
EOC_RE = re.compile(r"end_of_conversation.*?\*\*(true|false)\*\*", re.IGNORECASE)
CATALOG_URL_RE = re.compile(r"https?://www\.shl\.com/products/product-catalog/view/[^\s\)\|>]+")


@dataclass
class ParsedTurn:
    user_text: str
    agent_reply_expected: str
    urls_expected: List[str]
    eoc_expected: bool


def parse_trace(path: Path) -> List[ParsedTurn]:
    text = path.read_text(encoding="utf-8")
    turns_raw = TURN_HEADER_RE.split(text)[1:]  # drop preamble before first "### Turn"
    parsed: List[ParsedTurn] = []

    for block in turns_raw:
        user_match = USER_BLOCK_RE.search(block)
        agent_match = AGENT_BLOCK_RE.search(block)
        eoc_match = EOC_RE.search(block)

        if not user_match:
            continue

        user_text = user_match.group(1).strip()
        agent_text = agent_match.group(1).strip() if agent_match else ""
        urls = CATALOG_URL_RE.findall(agent_text)
        eoc = eoc_match.group(1).lower() == "true" if eoc_match else False

        parsed.append(
            ParsedTurn(
                user_text=user_text,
                agent_reply_expected=agent_text,
                urls_expected=urls,
                eoc_expected=eoc,
            )
        )
    return parsed


# ---------------------------------------------------------------------------
# 2. Load the real catalog so we can check "no hallucinated names" for real
# ---------------------------------------------------------------------------

def load_catalog_urls(catalog_path: Optional[Path]) -> Optional[set]:
    """If you have data/catalog.json available, pass it in to check
    every URL returned by the live system against the real catalog."""
    if not catalog_path or not catalog_path.exists():
        return None
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    return {item["url"].strip().rstrip("/") for item in data}


# ---------------------------------------------------------------------------
# 3. Replay each trace against the live API, turn by turn
# ---------------------------------------------------------------------------

@dataclass
class TraceResult:
    trace_id: str
    n_turns: int
    completed: bool = False
    schema_valid_turns: int = 0
    total_response_turns: int = 0
    recommendations_returned: int = 0
    catalog_grounded: int = 0
    catalog_ungrounded: int = 0
    eoc_correct_final_turn: Optional[bool] = None
    latencies_sec: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def run_trace(base_url: str, trace_id: str, turns: List[ParsedTurn],
              catalog_urls: Optional[set], timeout: int = 30) -> TraceResult:
    result = TraceResult(trace_id=trace_id, n_turns=len(turns))
    history: List[Dict[str, str]] = []

    try:
        for i, turn in enumerate(turns):
            history.append({"role": "user", "content": turn.user_text})

            start = time.monotonic()
            try:
                resp = requests.post(
                    f"{base_url.rstrip('/')}/chat",
                    json={"messages": history},
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                result.errors.append(f"turn {i+1}: request failed - {exc}")
                break
            elapsed = time.monotonic() - start
            result.latencies_sec.append(elapsed)

            if resp.status_code != 200:
                result.errors.append(f"turn {i+1}: HTTP {resp.status_code} - {resp.text[:200]}")
                break

            try:
                data = resp.json()
            except ValueError:
                result.errors.append(f"turn {i+1}: response not valid JSON")
                break

            # ---- schema check (mirrors app/schemas.py::ChatResponse) ----
            result.total_response_turns += 1
            required_keys = {"reply", "recommendations", "end_of_conversation"}
            if required_keys.issubset(data.keys()) and isinstance(data.get("reply"), str):
                result.schema_valid_turns += 1
            else:
                result.errors.append(f"turn {i+1}: schema mismatch, keys={list(data.keys())}")

            recs = data.get("recommendations") or []
            result.recommendations_returned += len(recs)

            if catalog_urls is not None:
                for rec in recs:
                    url = str(rec.get("url", "")).strip().rstrip("/")
                    if url in catalog_urls:
                        result.catalog_grounded += 1
                    else:
                        result.catalog_ungrounded += 1

            history.append({"role": "assistant", "content": data.get("reply", "")})

            if i == len(turns) - 1:
                actual_eoc = bool(data.get("end_of_conversation", False))
                result.eoc_correct_final_turn = (actual_eoc == turn.eoc_expected)

        else:
            result.completed = True
        # loop completed without an early "break" via errors
        if not result.errors:
            result.completed = True

    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"unhandled: {exc}")

    return result


# ---------------------------------------------------------------------------
# 4. Aggregate + report
# ---------------------------------------------------------------------------

def aggregate(results: List[TraceResult]) -> Dict[str, Any]:
    n = len(results)
    completed = sum(r.completed for r in results)
    schema_valid = sum(r.schema_valid_turns for r in results)
    schema_total = sum(r.total_response_turns for r in results)
    grounded = sum(r.catalog_grounded for r in results)
    ungrounded = sum(r.catalog_ungrounded for r in results)
    total_recs = sum(r.recommendations_returned for r in results)
    eoc_checks = [r for r in results if r.eoc_correct_final_turn is not None]
    eoc_correct = sum(1 for r in eoc_checks if r.eoc_correct_final_turn)
    all_latencies = [l for r in results for l in r.latencies_sec]

    summary = {
        "traces_evaluated": n,
        "traces_completed_end_to_end": completed,
        "completion_rate": f"{completed}/{n}",
        "schema_compliance": f"{schema_valid}/{schema_total}" if schema_total else "n/a",
        "total_recommendations_returned": total_recs,
        "catalog_grounding": (
            f"{grounded}/{grounded + ungrounded}" if (grounded + ungrounded) > 0
            else "not measured (no catalog.json supplied)"
        ),
        "hallucination_rate": (
            f"{ungrounded}/{grounded + ungrounded}" if (grounded + ungrounded) > 0
            else "not measured (no catalog.json supplied)"
        ),
        "end_of_conversation_flag_accuracy": f"{eoc_correct}/{len(eoc_checks)}" if eoc_checks else "n/a",
        "avg_latency_sec": round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else None,
        "p95_latency_sec": (
            round(sorted(all_latencies)[int(len(all_latencies) * 0.95) - 1], 2)
            if len(all_latencies) >= 5 else None
        ),
        "recall_at_10": "NOT REPORTED - no ground-truth relevance labels available in traces",
    }
    return summary


def write_markdown_report(summary: Dict[str, Any], results: List[TraceResult], out_path: Path) -> None:
    lines = ["# Evaluation Report\n"]
    lines.append("## Summary Metrics (computed from live /chat responses)\n")
    lines.append("| Evaluation Metric | Result |")
    lines.append("|---|---|")
    lines.append(f"| Traces Evaluated | {summary['traces_evaluated']} |")
    lines.append(f"| End-to-End Completion | {summary['completion_rate']} |")
    lines.append(f"| Schema Compliance | {summary['schema_compliance']} |")
    lines.append(f"| Catalog Grounding | {summary['catalog_grounding']} |")
    lines.append(f"| Hallucination Rate | {summary['hallucination_rate']} |")
    lines.append(f"| end_of_conversation Flag Accuracy | {summary['end_of_conversation_flag_accuracy']} |")
    lines.append(f"| Avg Latency | {summary['avg_latency_sec']} sec |")
    lines.append(f"| P95 Latency | {summary['p95_latency_sec']} sec |")
    lines.append(f"| Recall@10 | {summary['recall_at_10']} |")
    lines.append("\n## Per-Trace Detail\n")
    lines.append("| Trace | Turns | Completed | Schema OK | Recs Returned | Errors |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.trace_id} | {r.n_turns} | {'✅' if r.completed else '❌'} | "
            f"{r.schema_valid_turns}/{r.total_response_turns} | {r.recommendations_returned} | "
            f"{'; '.join(r.errors) if r.errors else '-'} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--traces-dir", required=True)
    parser.add_argument("--catalog", default=None, help="optional path to data/catalog.json for grounding check")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir)
    md_files = sorted(traces_dir.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {traces_dir}")

    catalog_urls = load_catalog_urls(Path(args.catalog)) if args.catalog else None
    if catalog_urls is None:
        print("[warn] No --catalog supplied: catalog grounding / hallucination rate will NOT be measured.")

    results: List[TraceResult] = []
    for md_file in md_files:
        turns = parse_trace(md_file)
        if not turns:
            print(f"[warn] {md_file.name}: no turns parsed, skipping")
            continue
        print(f"Running {md_file.stem} ({len(turns)} turns)...")
        result = run_trace(args.base_url, md_file.stem, turns, catalog_urls)
        results.append(result)
        status = "OK" if result.completed else "FAILED"
        print(f"  -> {status} | schema {result.schema_valid_turns}/{result.total_response_turns} "
              f"| recs {result.recommendations_returned} | errors: {result.errors}")

    summary = aggregate(results)

    out_dir = Path(args.out_dir)
    (out_dir / "eval_results.json").write_text(
        json.dumps({"summary": summary, "traces": [r.__dict__ for r in results]}, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(summary, results, out_dir / "eval_report.md")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nWrote eval_results.json and eval_report.md to {out_dir.resolve()}")


if __name__ == "__main__":
    main()