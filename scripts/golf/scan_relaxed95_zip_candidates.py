#!/usr/bin/env python3
"""Read-only discovery of cheaper task models stored inside submission ZIPs.

ZIP archives are never rewritten or extracted into the repository.  Candidate
bytes are hashed across archives, evaluated from a temporary file in an
isolated subprocess, and recorded for the same fresh-95% promotion gate used by
``scan_relaxed95_candidates.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path

from scan_relaxed95_candidates import (
    REPO,
    load_costs,
    load_prior_hashes,
    score_isolated,
)


TASK_RE = re.compile(r"^task(\d{1,3})\.onnx$", re.IGNORECASE)


def archives_from(sources: list[Path]) -> list[Path]:
    archives: set[Path] = set()
    for source in sources:
        if source.is_file() and source.suffix.lower() == ".zip":
            archives.add(source)
        elif source.is_dir():
            archives.update(source.rglob("*.zip"))
    return sorted(archives)


def load_recorded_digests(results_dir: Path) -> set[tuple[int, str]]:
    seen = load_prior_hashes(results_dir)
    for result_path in sorted(results_dir.glob("*.json")):
        try:
            rows = json.loads(result_path.read_text()).get("rows", [])
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            if row.get("task") and row.get("digest"):
                seen.add((int(row["task"]), str(row["digest"])))
    return seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--low", type=int, default=150)
    parser.add_argument("--high", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument(
        "--incumbents",
        type=Path,
        default=REPO / "artifacts" / "relaxed95_loop" / "incumbents.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = REPO / args.output
    if not args.incumbents.is_absolute():
        args.incumbents = REPO / args.incumbents

    sources = [path if path.is_absolute() else REPO / path for path in map(Path, args.sources)]
    archives = archives_from(sources)
    costs = load_costs(REPO / "all_scores.csv", args.incumbents)
    seen = load_recorded_digests(args.output.parent)
    prior_count = len(seen)
    rows: list[dict] = []
    scanned_members = 0
    invalid_archives = 0

    with tempfile.TemporaryDirectory(prefix="neurogolf-zip-scan-") as temp_dir:
        temp_path = Path(temp_dir) / "candidate.onnx"
        for archive_index, archive in enumerate(archives, 1):
            try:
                with zipfile.ZipFile(archive) as handle:
                    for member in handle.infolist():
                        match = TASK_RE.match(Path(member.filename).name)
                        if not match:
                            continue
                        task = int(match.group(1))
                        base_cost = costs.get(task)
                        if base_cost is None or not args.low <= base_cost <= args.high:
                            continue
                        scanned_members += 1
                        data = handle.read(member)
                        digest = hashlib.sha1(data).hexdigest()
                        key = (task, digest)
                        if key in seen:
                            continue
                        seen.add(key)
                        temp_path.write_bytes(data)
                        result = score_isolated(temp_path, task, args.timeout)
                        candidate_cost = result.get("cost")
                        row = {
                            "task": task,
                            "base_cost": base_cost,
                            "candidate_cost": candidate_cost,
                            "correct": result.get("correct", False),
                            "archive": str(archive.relative_to(REPO)),
                            "member": member.filename,
                            "digest": digest,
                        }
                        if result.get("error"):
                            row["error"] = result["error"]
                        if (
                            result.get("correct")
                            and isinstance(candidate_cost, int)
                            and candidate_cost < base_cost
                        ):
                            row["gain"] = math.log(base_cost / candidate_cost)
                            print(
                                f"WIN task{task:03d} {base_cost}->{candidate_cost} "
                                f"+{row['gain']:.6f} {row['archive']}::{member.filename}",
                                flush=True,
                            )
                        rows.append(row)
                        if len(rows) % 20 == 0:
                            print(
                                f"progress candidates={len(rows)} "
                                f"archives={archive_index}/{len(archives)}",
                                flush=True,
                            )
            except (OSError, zipfile.BadZipFile, RuntimeError):
                invalid_archives += 1

    winners = sorted(
        (row for row in rows if "gain" in row),
        key=lambda row: (-row["gain"], row["task"]),
    )
    payload = {
        "policy": {"fresh_rate": 0.95, "fresh_k": 5000},
        "range": [args.low, args.high],
        "sources": [str(path.relative_to(REPO)) for path in sources],
        "archive_count": len(archives),
        "invalid_archives": invalid_archives,
        "prior_hashes": prior_count,
        "scanned_members": scanned_members,
        "candidate_count": len(rows),
        "winners": winners,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"archives={len(archives)} members={scanned_members} "
        f"candidates={len(rows)} winners={len(winners)} "
        f"visible_gain=+{sum(row['gain'] for row in winners):.6f}",
        flush=True,
    )
    print(f"output={args.output.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
