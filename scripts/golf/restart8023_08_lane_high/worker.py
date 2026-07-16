#!/usr/bin/env python3
"""8023.08 evidence-only high-cost/new-drop search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OLD = import_path(
    "restart8023_08_high_old",
    ROOT / "scripts/golf/restart8019_75_lane_high/worker.py",
)
BASE = OLD.BASE

AUTHORITY = ROOT / "submission_base_8023.08.zip"
AUTHORITY_SHA256 = "0e29e8d57f7ac58136a9574351c9c6f3056f9debf6eeee9c181c8f2e9fac690a"
NEW_DIRS = (ROOT / "others/71609",)
EXCLUDED = {132, 168, 226, 275, 345}


def current_band() -> tuple[tuple[int, int], ...]:
    rows: list[tuple[int, int]] = []
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"][4:])
            cost = int(row["cost"])
            if cost >= 300 and task not in EXCLUDED:
                rows.append((task, cost))
    return tuple(sorted(rows, key=lambda item: (-item[1], item[0])))


BAND = current_band()
ELIGIBLE = tuple(task for task, _cost in BAND)
COSTS = dict(BAND)


def configure() -> None:
    OLD.HERE = HERE
    OLD.AUTHORITY = AUTHORITY
    OLD.AUTHORITY_SHA256 = AUTHORITY_SHA256
    OLD.NEW_DIRS = NEW_DIRS
    OLD.BAND = BAND
    OLD.ELIGIBLE = ELIGIBLE
    OLD.COSTS = COSTS

    BASE.HERE = HERE
    BASE.AUTHORITY = AUTHORITY
    BASE.AUTHORITY_SHA256 = AUTHORITY_SHA256
    BASE.BAND = BAND
    BASE.ELIGIBLE = ELIGIBLE
    BASE.COSTS = COSTS
    BASE.PRIVATE_ZERO_OR_UNSOUND = set()
    BASE.EXPLICIT_LATEST_LB_BLACK = set()
    BASE.CHANGED_FROM_8011_05 = set(ELIGIBLE)
    BASE.THRESHOLD = 1.0
    BASE.FRESH_PER_SEED = 2_000
    BASE.SUPPORT.POLICY_THRESHOLD = 1.0
    BASE.SUPPORT.FRESH_PER_SEED = 2_000
    BASE.SUPPORT.evaluate_four = OLD.exact_failfast_evaluate


configure()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_worker(worker_id: int) -> Any:
    worker = BASE.Worker(worker_id)
    OLD.reprofile_authority(worker)
    return worker


def screen(worker: Any) -> None:
    OLD.scan_new_drops(worker)
    worker.scan_current_simplifiers()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if digest(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("8023.08 authority SHA mismatch")

    worker = make_worker(args.worker)
    screen(worker)
    pre = worker.full_audit()
    gates = []
    accepted = []
    for row in pre:
        path = ROOT / row["saved_path"]
        gate = OLD.strict_gate(path, int(row["task"]), int(row["authority_cost"]))
        gates.append(gate)
        if gate["pass"]:
            row["strict_gate"] = gate
            accepted.append(row)
    payload = {
        "worker": args.worker,
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "assigned_tasks": list(worker.tasks),
        "band_count": len(BAND),
        "task_rows": [worker.task_rows[task] for task in worker.tasks],
        "counters": dict(worker.counters),
        "pre_strict_finalists": pre,
        "strict_gates": gates,
        "finalists": accepted,
        "excluded": sorted(EXCLUDED),
        "absolute_gate": (
            "local+official gold exact, strict/static, stable margin, "
            "fresh2000x2 100%, zero runtime/nonfinite/shape/small-positive"
        ),
        "protected_writes": "lane only; root authority unchanged",
    }
    output = HERE / f"worker_{args.worker}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "worker": args.worker,
        "tasks": len(worker.tasks),
        "strict_winners": [
            {
                "task": row["task"],
                "cost": row["candidate_cost"],
                "gain": row["score_gain"],
                "path": row["saved_path"],
            }
            for row in accepted
        ],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
