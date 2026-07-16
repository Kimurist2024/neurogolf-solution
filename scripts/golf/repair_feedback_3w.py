#!/usr/bin/env python3
"""Run the evidence-only low-cost repair worker on a feedback band."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTHORITY = ROOT / "submission_base_8023.08.zip"
AUTHORITY_SHA256 = "0e29e8d57f7ac58136a9574351c9c6f3056f9debf6eeee9c181c8f2e9fac690a"
BLACK = {
    9, 66, 70, 76, 93, 105, 134, 185, 187, 198, 199, 201, 202, 208,
    219, 233, 277, 286, 343, 365, 366, 396,
}
PRIVATE_ZERO = {
    9, 15, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    118, 133, 134, 138, 145, 157, 158, 168, 169, 170, 173, 174, 178, 182,
    185, 187, 191, 192, 196, 198, 202, 204, 205, 208, 209, 216, 219, 222,
    233, 246, 251, 255, 273, 277, 285, 286, 302, 319, 325, 333, 343, 346,
    361, 365, 366, 372, 377, 379, 391, 393, 396,
}
ADOPTED = {23, 354}
FAILED_GOLD = {12, 110, 161, 175, 188, 355}
FOCUS_WHITE_OVERRIDE = {102, 168, 175, 192, 205, 219, 286, 354}


def import_worker():
    path = ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py"
    spec = importlib.util.spec_from_file_location("feedback_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type=int, required=True)
    parser.add_argument("--high", type=int, required=True)
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="*", default=None)
    args = parser.parse_args()
    module = import_worker()
    module.AUTHORITY = AUTHORITY
    if hashlib.sha256(module.AUTHORITY.read_bytes()).hexdigest() != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA drift")
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        import csv
        costs = {
            int(row["task"].removeprefix("task")): int(row["cost"])
            for row in csv.DictReader(handle)
        }
    # Reprofile the current authority instead of trusting legacy CSV costs.
    # Some accepted graphs have truthful runtime costs that differ from the
    # historical ledger; using the ledger made the worker abort before search.
    actual_costs = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in sorted(costs):
            data = archive.read(f"task{task:03d}.onnx")
            model = module.onnx.load_model_from_string(data)
            known_cases, _ = module.SUPPORT.known_cases(task)
            profile = module.POLICY.fast_profile(module.SUPPORT, task, model, known_cases[0])
            if profile is not None:
                actual_costs[task] = int(profile["cost"])
    band = tuple(
        (task, cost) for task, cost in sorted(actual_costs.items())
        if args.low <= cost <= args.high
        and (not args.tasks or task in set(args.tasks))
    )
    module.BAND = band
    module.COSTS = dict(band)
    module.PRIVATE_ZERO_OR_UNSOUND = (
        PRIVATE_ZERO | BLACK | ADOPTED | FAILED_GOLD
    ) - FOCUS_WHITE_OVERRIDE
    module.ELIGIBLE = tuple(task for task, _ in band if task not in module.PRIVATE_ZERO_OR_UNSOUND)
    module.CHANGED_FROM_8011_05 = set()
    module.HERE = args.output
    # Gold is a hard gate: fresh/policy accuracy alone is never sufficient.
    module.THRESHOLD = 1.0
    module.FRESH_PER_SEED = 2_000
    module.POLICY.POLICY_THRESHOLD = 0.90
    module.SUPPORT.FRESH_PER_SEED = 2_000
    args.output.mkdir(parents=True, exist_ok=True)
    payload = module.Worker(args.worker).run()
    # Re-run the repository's authoritative train/test/arc-gen gold checker on
    # every finalist.  Candidates failing gold are removed from both the
    # admission list and the on-disk candidate directory.
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
        row["official_gold"] = check.returncode == 0
        row["official_gold_output"] = (check.stdout + check.stderr)[-2000:]
        if check.returncode == 0:
            admitted.append(row)
        else:
            path.unlink(missing_ok=True)
    payload["finalists"] = admitted
    payload["gold_required"] = True
    destination = args.output / f"worker_{args.worker}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "worker": args.worker,
        "band": [args.low, args.high],
        "assigned": payload["assigned_tasks"],
        "finalists": [
            {"task": row["task"], "cost": row["candidate_cost"], "gain": row["score_gain"]}
            for row in payload["finalists"]
        ],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
