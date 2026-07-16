#!/usr/bin/env python3
"""Rebase every retained historical model that can halve a cost<=500 task."""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import onnx
import onnxruntime


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
EVIDENCE = HERE / "evidence.json"


def load_base():
    path = ROOT / "scripts/golf/root_cost50_history_scan_298/scan.py"
    spec = importlib.util.spec_from_file_location("history_scan_308_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def current_costs() -> dict[int, int]:
    result = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"].removeprefix("task"))
            cost = int(row["cost"])
            if cost <= 500 and float(row["score"]) < 25.0:
                result[task] = cost
    return result


def evaluate(base, index: int, row: dict[str, object], incumbent: int) -> dict[str, object]:
    task = int(row["task"])
    path = ROOT / str(row["path"])
    model = onnx.load(path)
    try:
        fast_exact, fast_checked = base.known_exact_fast(model, task)
    except Exception:
        fast_exact, fast_checked = False, 0
    profile = None
    if fast_exact:
        try:
            with tempfile.TemporaryDirectory(prefix=f"hist308_{task:03d}_", dir="/tmp") as work:
                profile = base.scoring.score_and_verify(
                    model, task, work, label=f"hist{index}", require_correct=False
                )
        except Exception:
            profile = None
    item = dict(row)
    item["index"] = index
    item["fast_known_exact"] = fast_exact
    item["fast_known_checked"] = fast_checked
    item["profile"] = profile
    item["known_exact"] = bool(profile is not None and profile["correct"])
    item["half_cost_actual"] = bool(
        item["known_exact"] and int(profile["cost"]) * 2 <= incumbent
    )
    return item


def main() -> int:
    started = time.monotonic()
    base = load_base()
    onnxruntime.set_default_logger_severity(3)
    authority_sha256 = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    costs = current_costs()
    raw_paths = subprocess.check_output(
        ["rg", "--files", "-g", "*.onnx"], cwd=ROOT, text=True
    ).splitlines()

    seen: set[tuple[int, str]] = set()
    candidates: list[dict[str, object]] = []
    for relpath in raw_paths:
        task = base.task_from_path(relpath)
        if task not in costs:
            continue
        path = ROOT / relpath
        try:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            key = (task, digest)
            if key in seen:
                continue
            seen.add(key)
            model = onnx.load_model_from_string(data)
        except Exception:
            continue
        params = base.parameter_count(model)
        lower_bound = base.declared_lower_bound(model)
        # Both are necessary lower bounds. If either exceeds half, the actual
        # validator cost cannot satisfy this campaign's admission target.
        if params * 2 > costs[task] or lower_bound * 2 > costs[task]:
            continue
        candidates.append({
            "task": task,
            "path": relpath,
            "sha256": digest,
            "authority_cost": costs[task],
            "params": params,
            "declared_lower_bound": lower_bound,
            "node_count": len(model.graph.node),
            "ops": [node.op_type for node in model.graph.node],
        })

    workers = max(1, int(os.environ.get("NG_HISTORY_WORKERS", "6")))
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluate, base, index, row, costs[int(row["task"])]): (index, row)
            for index, row in enumerate(candidates, start=1)
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            index, row = futures[future]
            try:
                item = future.result()
            except Exception:
                item = dict(row)
                item.update({
                    "index": index,
                    "fast_known_exact": False,
                    "fast_known_checked": 0,
                    "profile": None,
                    "known_exact": False,
                    "half_cost_actual": False,
                })
            results.append(item)
            completed += 1
            profile = item["profile"]
            print(json.dumps({
                "completed": completed,
                "total": len(candidates),
                "index": index,
                "task": item["task"],
                "cost": None if profile is None else profile["cost"],
                "correct": None if profile is None else profile["correct"],
                "half": item["half_cost_actual"],
            }), flush=True)

    results.sort(key=lambda row: int(row["index"]))
    winners = [row for row in results if row["half_cost_actual"]]
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": authority_sha256,
        "authority_lb": 8011.05,
        "scope": "all 322 non-score25 authority tasks with cost<=500",
        "target": "actual candidate cost <= half authority cost",
        "path_count": len(raw_paths),
        "unique_task_sha_count": len(seen),
        "theoretical_half_candidates": len(candidates),
        "workers": workers,
        "known_exact_half_winners": len(winners),
        "half_cost_winners": winners,
        "winners": winners,
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
        "protected_writes": "none; only this evidence directory",
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidates": len(candidates),
        "winners": len(winners),
        "evidence": str(EVIDENCE.relative_to(ROOT)),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
