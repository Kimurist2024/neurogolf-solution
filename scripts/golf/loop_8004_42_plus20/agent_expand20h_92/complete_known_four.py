#!/usr/bin/env python3
"""Run known gold in all four ORT/thread configurations for every lower job."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
from screen_all import resolve_source  # noqa: E402
from harvest import run_bounded  # noqa: E402

SPEC = importlib.util.spec_from_file_location("expand20h92_probe_helpers", HERE / "rescreen_probe_candidates.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load known4 worker")
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def known4_worker(job):
    digest, task, data = job
    try:
        return {"sha256": digest, "task": task, "known_four": HELPERS.SCAN.known_four(task, data)}
    except BaseException as exc:  # noqa: BLE001
        return {"sha256": digest, "task": task, "error": f"{type(exc).__name__}: {exc}"}


def resolve(row):
    for source in row["sources"]:
        data = resolve_source(source, int(row["task"]))
        if data is not None and hashlib.sha256(data).hexdigest() == row["sha256"]:
            return data, source
    raise RuntimeError(f"unresolved {row['sha256']}")


def main() -> int:
    rows = json.loads((HERE / "rescreen.json").read_text())["rows"]
    stages = {"known_reject", "known_dual_reject", "shape_reject", "fresh500_reject", "fresh500_pass"}
    selected = [row for row in rows if row["stage"] in stages]
    jobs = []
    metadata = {}
    for row in selected:
        data, source = resolve(row)
        jobs.append((row["sha256"], int(row["task"]), data))
        metadata[row["sha256"]] = {
            "task": row["task"], "sha256": row["sha256"], "source": source,
            "stage": row["stage"], "actual_screen_cost": row.get("actual_screen_cost"),
            "official_like_score": row.get("official_like_score"),
        }
    results = run_bounded(jobs, known4_worker, max_workers=4, timeout=60.0, label="KNOWN4_ALL")
    output = []
    for result in results:
        digest = result.get("sha256")
        item = {**metadata.get(digest, {}), **result}
        quad = item.get("known_four")
        item["known_four_complete"] = HELPERS.known_complete(quad)
        output.append(item)
    output.sort(key=lambda item: (item.get("task", 999), item.get("sha256", "")))
    report = {
        "expected_count": len(selected), "result_count": len(output),
        "complete_known_four_count": sum(bool(item["known_four_complete"]) for item in output),
        "rows": output,
    }
    (HERE / "audit/known_four_all_actual_lower.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("expected_count", "result_count", "complete_known_four_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
