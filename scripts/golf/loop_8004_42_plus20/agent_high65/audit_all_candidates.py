#!/usr/bin/env python3
"""SHA-deduplicated audit of all loose ONNX history for high65 targets."""

from __future__ import annotations

import copy
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "submission_base_8005.16.zip"
OUT = Path(__file__).with_name("all_candidate_audit.json")
TARGETS = (276, 305, 309, 312, 337, 373, 53, 87)

import sys

sys.path.insert(0, str(ROOT))
from scripts.golf.loop_8004_42_plus20.audit_retained_group import (  # noqa: E402
    perfect,
    profile,
    sha,
)
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import (  # noqa: E402
    run_known,
    structure,
)


def main() -> None:
    baseline_data: dict[int, bytes] = {}
    baseline_cost: dict[int, int] = {}
    baseline_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(BASE) as archive:
        for task in TARGETS:
            member = f"task{task:03d}.onnx"
            data = archive.read(member)
            actual = profile(data, member)
            baseline_data[task] = data
            baseline_cost[task] = int(actual["cost"])
            baseline_rows.append(
                {
                    "task": task,
                    "sha256": sha(data),
                    "actual_cost": actual,
                }
            )

    # Keep all aliases as coverage evidence, but run cost/known only once per task/SHA.
    aliases: dict[tuple[int, str], list[str]] = defaultdict(list)
    payloads: dict[tuple[int, str], bytes] = {}
    for task in TARGETS:
        for path in ROOT.rglob(f"task{task:03d}*.onnx"):
            if not path.is_file():
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            digest = sha(data)
            key = (task, digest)
            aliases[key].append(str(path.relative_to(ROOT)))
            payloads.setdefault(key, data)

    rows: list[dict[str, object]] = []
    for (task, digest), data in sorted(payloads.items()):
        paths = sorted(set(aliases[(task, digest)]))
        actual = profile(data, Path(paths[0]).name)
        cost = actual.get("cost")
        lower = cost is not None and int(cost) < baseline_cost[task]
        row: dict[str, object] = {
            "task": task,
            "representative_path": paths[0],
            "alias_count": len(paths),
            "aliases": paths,
            "sha256": digest,
            "same_as_baseline": data == baseline_data[task],
            "actual_cost": actual,
            "baseline_actual_cost": baseline_cost[task],
            "strictly_cheaper": lower,
        }
        if lower:
            try:
                model = onnx.load_model_from_string(data)
                disable = run_known(copy.deepcopy(model), task, True)
                default = (
                    run_known(copy.deepcopy(model), task, False)
                    if perfect(disable)
                    else {"not_run": "disable_all_not_perfect"}
                )
                dual = perfect(disable) and perfect(default)
                row["known_disable_all"] = disable
                row["known_default"] = default
                row["structure"] = structure(copy.deepcopy(model), task) if dual else None
                row["promising_pre_fresh"] = dual
                print(
                    f"LOWER task{task:03d} cost={cost}/{baseline_cost[task]} "
                    f"known={disable.get('right')}/{disable.get('total')} aliases={len(paths)}",
                    flush=True,
                )
            except Exception as exc:
                row["audit_error"] = f"{type(exc).__name__}: {exc}"
                row["promising_pre_fresh"] = False
        rows.append(row)

    counts: dict[str, dict[str, int]] = {}
    for task in TARGETS:
        selected = [row for row in rows if row["task"] == task]
        counts[str(task)] = {
            "path_aliases": sum(int(row["alias_count"]) for row in selected),
            "unique_sha": len(selected),
            "nonbaseline_sha": sum(not bool(row["same_as_baseline"]) for row in selected),
            "strict_lower": sum(bool(row["strictly_cheaper"]) for row in selected),
            "known100_dual_strict_lower": sum(bool(row.get("promising_pre_fresh")) for row in selected),
        }

    result = {
        "baseline": {"path": BASE.name, "sha256": sha(BASE.read_bytes())},
        "targets": list(TARGETS),
        "baseline_rows": baseline_rows,
        "coverage_counts": counts,
        "unique_task_model_pairs": len(rows),
        "path_aliases": sum(len(v) for v in aliases.values()),
        "candidate_rows": rows,
        "lower_rows": [row for row in rows if row["strictly_cheaper"]],
        "promising": [
            {
                "task": row["task"],
                "path": row["representative_path"],
                "sha256": row["sha256"],
            }
            for row in rows
            if row.get("promising_pre_fresh")
        ],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
