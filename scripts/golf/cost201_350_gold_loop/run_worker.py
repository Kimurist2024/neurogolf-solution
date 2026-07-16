#!/usr/bin/env python3
"""Run one cost-201..350 search worker against the 8012.23 authority.

The score ledger contains a few legacy declared-cost values that differ from
the current scorer's reprofile of the immutable authority member.  This lane
uses the freshly measured authority cost as the comparison floor, so a stale
ledger value can never manufacture a false gain.  Admission still requires
the repository's official gold checker after the worker's exact-known and
fresh audits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.23.zip"
AUTHORITY_SHA256 = "720ebf75d826945250e3c7d7ea11780a950d8d3038546e9c7595503277a1189f"
EXCLUDED = {
    9, 12, 15, 23, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 96, 101,
    102, 110, 112, 118, 133, 134, 138, 145, 157, 158, 161, 168, 169,
    170, 173, 174, 175, 178, 182, 185, 187, 188, 191, 192, 196, 198,
    202, 204, 205, 208, 209, 216, 219, 222, 233, 246, 251, 255, 273,
    277, 285, 286, 302, 319, 325, 333, 343, 346, 354, 355, 361, 365,
    366, 372, 377, 379, 391, 393, 396,
}


def import_worker():
    path = ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py"
    spec = importlib.util.spec_from_file_location("cost201_350_base_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()

    module = import_worker()
    module.AUTHORITY = AUTHORITY
    if hashlib.sha256(AUTHORITY.read_bytes()).hexdigest() != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA drift")

    import csv
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        ledger = {
            int(row["task"].removeprefix("task")): int(row["cost"])
            for row in csv.DictReader(handle)
        }
    band = tuple(
        (task, cost) for task, cost in sorted(ledger.items())
        if 201 <= cost <= 350
    )
    module.BAND = band
    module.COSTS = dict(band)
    module.PRIVATE_ZERO_OR_UNSOUND = set(EXCLUDED)
    module.ELIGIBLE = tuple(task for task, _ in band if task not in EXCLUDED)
    module.CHANGED_FROM_8011_05 = set()
    module.HERE = HERE
    module.THRESHOLD = 1.0
    module.FRESH_PER_SEED = 2_000
    module.SUPPORT.FRESH_PER_SEED = 2_000

    def reprofile_authority(worker) -> None:
        with zipfile.ZipFile(AUTHORITY) as archive:
            for task in worker.tasks:
                data = archive.read(f"task{task:03d}.onnx")
                model = onnx.load_model_from_string(data)
                cases, _ = worker.cases[task]
                profile = module.POLICY.fast_profile(module.SUPPORT, task, model, cases[0])
                if profile is None:
                    raise RuntimeError(f"task{task:03d} authority did not profile")
                ledger_cost = int(module.COSTS[task])
                measured_cost = int(profile["cost"])
                module.COSTS[task] = measured_cost
                worker.task_rows[task]["authority_cost"] = measured_cost
                worker.task_rows[task]["authority_ledger_cost"] = ledger_cost
                worker.task_rows[task]["authority_reprofile"] = {
                    "sha256": module.digest(data),
                    "profile": profile,
                    "ledger_cost": ledger_cost,
                    "measured_cost": measured_cost,
                    "comparison_cost": measured_cost,
                    "cost_matches_census": measured_cost == ledger_cost,
                }
                worker.seen[task].add(module.digest(data))

    module.Worker.reprofile_authority = reprofile_authority
    payload = module.Worker(args.worker).run()

    admitted = []
    for row in payload.get("finalists", []):
        saved = row.get("saved_path")
        if not saved:
            continue
        path = ROOT / saved
        task = int(row["task"])
        check = subprocess.run(
            [sys.executable, str(ROOT / "scripts/golf/try_candidate.py"),
             "--task", str(task), "--onnx", str(path)],
            cwd=ROOT, capture_output=True, text=True,
        )
        output = check.stdout + check.stderr
        row["official_gold"] = check.returncode == 0 and "PASS gold:" in output
        row["official_gold_output"] = output[-4000:]
        if row["official_gold"]:
            admitted.append(row)
        else:
            path.unlink(missing_ok=True)
    payload["finalists"] = admitted
    payload["gold_required"] = True
    payload["authority_cost_policy"] = "fresh scorer reprofile overrides stale ledger"
    destination = HERE / f"worker_{args.worker}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "worker": args.worker,
        "assigned": payload["assigned_tasks"],
        "finalists": [
            {"task": row["task"], "cost": row["candidate_cost"],
             "gain": row["score_gain"]}
            for row in admitted
        ],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
