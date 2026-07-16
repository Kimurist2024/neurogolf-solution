#!/usr/bin/env python3
"""Worker 1: reprice every retained historical cost<=166 task model."""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import onnx

import common


OUT = common.HERE / "history_scan.json"


def evaluate(index: int, row: dict[str, object]) -> dict[str, object]:
    task = int(row["task"])
    model = onnx.load(common.ROOT / str(row["path"]))
    try:
        fast, checked = common.HISTORY.known_exact_fast(model, task)
    except Exception:
        fast, checked = False, 0
    profile = None
    if fast:
        try:
            with tempfile.TemporaryDirectory(prefix=f"low405_hist_{task:03d}_", dir="/tmp") as work:
                profile = common.HISTORY.scoring.score_and_verify(
                    model, task, work, label=f"hist{index}", require_correct=False
                )
        except Exception:
            profile = None
    item = dict(row)
    item.update({"index": index, "fast_known_exact": fast, "fast_known_checked": checked, "profile": profile})
    item["known_exact"] = bool(profile is not None and profile["correct"])
    item["strict_lower_actual"] = bool(
        item["known_exact"] and int(profile["cost"]) < int(row["authority_cost"])
    )
    item["half_cost_actual"] = bool(
        item["known_exact"] and int(profile["cost"]) * 2 <= int(row["authority_cost"])
    )
    if item["strict_lower_actual"]:
        item["structure"] = common.PATTERN.structure(model)
    return item


def main() -> int:
    started = time.monotonic()
    common.HERE.mkdir(parents=True, exist_ok=True)
    common.CANDIDATES.mkdir(parents=True, exist_ok=True)
    common.validate_authority()
    costs = common.current_costs()
    paths = subprocess.check_output(["rg", "--files", "-g", "*.onnx"], cwd=common.ROOT, text=True).splitlines()
    seen: set[tuple[int, str]] = set()
    rows: list[dict[str, object]] = []
    for rel in paths:
        task = common.HISTORY.task_from_path(rel)
        if task not in costs:
            continue
        path = common.ROOT / rel
        try:
            data = path.read_bytes()
            digest = common.sha256(data)
            if (task, digest) in seen:
                continue
            seen.add((task, digest))
            model = onnx.load_model_from_string(data)
            params = common.HISTORY.parameter_count(model)
            lower = common.HISTORY.declared_lower_bound(model)
        except Exception:
            continue
        if params >= costs[task] or lower >= costs[task]:
            continue
        rows.append({
            "task": task, "path": rel, "sha256": digest,
            "authority_cost": costs[task], "params": params,
            "declared_lower_bound": lower, "node_count": len(model.graph.node),
            "ops": [node.op_type for node in model.graph.node],
        })

    # This process is one of three campaign workers.  It intentionally uses one
    # evaluation thread so the campaign has exactly three concurrent executors.
    workers = max(1, int(os.environ.get("NG_LOW_HISTORY_THREADS", "1")))
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(evaluate, i, row): (i, row) for i, row in enumerate(rows, 1)}
        for completed, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            i, row = future_map[future]
            try:
                item = future.result()
            except Exception as exc:  # noqa: BLE001
                item = dict(row)
                item.update({
                    "index": i, "fast_known_exact": False, "fast_known_checked": 0,
                    "profile": None, "known_exact": False, "strict_lower_actual": False,
                    "half_cost_actual": False, "worker_error": f"{type(exc).__name__}:{exc}",
                })
            results.append(item)
            print(json.dumps({
                "completed": completed, "total": len(rows), "task": item["task"],
                "cost": None if item["profile"] is None else item["profile"]["cost"],
                "strict": item["strict_lower_actual"], "half": item["half_cost_actual"],
            }), flush=True)
    results.sort(key=lambda row: int(row["index"]))
    strict = [row for row in results if row["strict_lower_actual"]]
    clean = [row for row in strict if row.get("structure", {}).get("pass")]
    payload = {
        "authority": str(common.AUTHORITY.relative_to(common.ROOT)),
        "authority_sha256": common.AUTHORITY_SHA256,
        "authority_lb": 8012.15,
        "authority_diff": common.authority_diff(),
        "scope_task_count": len(costs),
        "loose_onnx_paths": len(paths),
        "unique_task_sha_pairs": len(seen),
        "theoretical_strict_lower_candidates": len(rows),
        "known_exact_strict_lower_rows": len(strict),
        "structurally_clean_strict_lower_rows": len(clean),
        "known_exact_half_rows": sum(bool(row["half_cost_actual"]) for row in results),
        "strict_winners": strict,
        "structurally_clean_winners": clean,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
        "protected_writes": "lane only; authority/root/others untouched",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"strict": len(strict), "clean": len(clean), "out": str(OUT.relative_to(common.ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

