"""
Convert raw SHL-scraped catalog (entity_id/link/job_levels/keys format)
into the CatalogItem schema this app actually uses
(name/url/test_type/description/level/duration_minutes/remote_testing).

Usage:
    python scripts/convert_catalog.py data/catalog_raw.json data/catalog.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# --- mapping tables -----------------------------------------------------

# SHL "keys" -> our TestType enum. A raw item can have multiple keys;
# we pick the first one that maps, in this priority order.
KEY_TO_TEST_TYPE_PRIORITY = [
    ("Simulations", "coding"),
    ("Ability & Aptitude", "cognitive"),
    ("Knowledge & Skills", "skills"),
    ("Biodata & Situational Judgment", "situational_judgement"),
    ("Personality & Behavior", "personality"),
    ("Competencies", "behavioral"),
    ("Assessment Exercises", "behavioral"),
    ("Development & 360", "behavioral"),
]

# SHL job_levels -> our Level enum. Priority order matters (first match wins
# per raw level string).
JOB_LEVEL_TO_LEVEL = {
    "entry-level": "entry",
    "graduate": "graduate",
    "mid-professional": "mid",
    "professional individual contributor": "mid",
    "front line manager": "mid",
    "supervisor": "mid",
    "manager": "senior",
    "director": "senior",
    "executive": "senior",
    "general population": "all_levels",
}


def guess_test_type(keys: list[str]) -> str:
    for wanted_key, test_type in KEY_TO_TEST_TYPE_PRIORITY:
        if wanted_key in keys:
            return test_type
    return "skills"  # safe default — most SHL items are knowledge/skills tests


def guess_level(job_levels: list[str]) -> str:
    for jl in job_levels:
        mapped = JOB_LEVEL_TO_LEVEL.get(jl.strip().lower())
        if mapped:
            return mapped
    return "all_levels"


def parse_duration_minutes(duration_raw: str, duration: str) -> int | None:
    # Prefer the raw field ("Approximate Completion Time in minutes = 30")
    source = duration_raw or duration or ""
    match = re.search(r"(\d+)\s*$", source.strip())
    if match:
        return int(match.group(1))
    # Fallback: parse plain "30 minutes"
    match = re.search(r"(\d+)", duration or "")
    if match:
        return int(match.group(1))
    return None


def convert_item(raw: dict) -> dict | None:
    name = raw.get("name")
    url = raw.get("link")
    if not name or not url:
        return None  # skip malformed entries

    keys = raw.get("keys") or []
    job_levels = raw.get("job_levels") or []

    return {
        "name": name,
        "url": url,
        "test_type": guess_test_type(keys),
        "description": (raw.get("description") or "").strip() or name,
        "level": guess_level(job_levels),
        "duration_minutes": parse_duration_minutes(
            raw.get("duration_raw", ""), raw.get("duration", "")
        ),
        "remote_testing": (raw.get("remote") or "").strip().lower() == "yes",
    }


def main(src_path: str, dst_path: str) -> None:
    src = Path(src_path)
    dst = Path(dst_path)

    raw_items = json.loads(src.read_text())
    converted = []
    skipped = 0

    for raw in raw_items:
        item = convert_item(raw)
        if item is None:
            skipped += 1
            continue
        converted.append(item)

    dst.write_text(json.dumps(converted, indent=2, ensure_ascii=False))
    print(f"Converted {len(converted)} items -> {dst}")
    print(f"Skipped {skipped} malformed items")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/convert_catalog.py <raw_catalog.json> <output_catalog.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])