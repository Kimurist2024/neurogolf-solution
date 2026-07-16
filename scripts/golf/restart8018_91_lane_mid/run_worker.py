#!/usr/bin/env python3
"""8018.91 evidence-only history/exact scan for ledger cost 250..399.

This wrapper reuses the mature three-family scanner while pinning every
comparison to the immutable 8018.91 authority.  It never calls
``try_candidate.py`` (which can promote into the root tree); final local-gold
verification is performed by the non-mutating timeout verifier instead.
"""

from __future__ import annotations

import argparse
import csv
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
AUTHORITY = ROOT / "submission_base_8018.91.zip"
AUTHORITY_SHA256 = "e43865760ec8807fbb217fba718226ca6b86d9128b911479214e3252b9f9e091"

# Union of the maintained private-zero/unsound catalogue and the broader
# historical quarantine used by the mature scanner.  Exact known/fresh
# testing is still applied to every eligible model; this exclusion prevents
# spending the deadline on candidates already disproved by leaderboard data.
EXCLUDED = {
    9, 12, 15, 23, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 96, 101,
    102, 110, 112, 118, 133, 134, 138, 145, 157, 158, 161, 168, 169,
    170, 173, 174, 175, 178, 182, 185, 187, 188, 191, 192, 196, 198,
    202, 204, 205, 208, 209, 216, 219, 222, 233, 246, 251, 255, 273,
    277, 285, 286, 302, 319, 325, 333, 343, 346, 354, 355, 361, 365,
    366, 372, 377, 379, 391, 393, 396,
}


def load_base():
    path = ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py"
    spec = importlib.util.spec_from_file_location("restart8018_mid_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ledger_band() -> tuple[tuple[int, int], ...]:
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        return tuple(sorted(
            (int(row["task"].removeprefix("task")), int(row["cost"]))
            for row in csv.DictReader(handle)
            if 250 <= int(row["cost"]) <= 399
        ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    authority_sha = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    if authority_sha != AUTHORITY_SHA256:
        raise RuntimeError(f"authority SHA drift: {authority_sha}")

    module = load_base()
    band = ledger_band()
    module.AUTHORITY = AUTHORITY
    module.AUTHORITY_SHA256 = AUTHORITY_SHA256
    module.BAND = band
    module.COSTS = dict(band)
    module.PRIVATE_ZERO_OR_UNSOUND = set(EXCLUDED)
    module.ELIGIBLE = tuple(task for task, _ in band if task not in EXCLUDED)
    module.CHANGED_FROM_8011_05 = set()
    module.HERE = HERE
    module.THRESHOLD = 1.0
    module.FRESH_PER_SEED = 2_000
    module.SUPPORT.POLICY_THRESHOLD = 1.0
    module.SUPPORT.FRESH_PER_SEED = 2_000

    # Reprofile the immutable authority before any comparison.  This makes the
    # measured scorer cost, rather than a possibly stale CSV entry, the floor.
    def reprofile_authority(worker) -> None:
        with zipfile.ZipFile(AUTHORITY) as archive:
            for task in worker.tasks:
                data = archive.read(f"task{task:03d}.onnx")
                model = onnx.load_model_from_string(data)
                cases, _ = worker.cases[task]
                profile = module.POLICY.fast_profile(module.SUPPORT, task, model, cases[0])
                if profile is None:
                    raise RuntimeError(f"task{task:03d}: authority profile failed")
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
                }
                worker.seen[task].add(module.digest(data))

    module.Worker.reprofile_authority = reprofile_authority
    payload = module.Worker(args.worker).run()

    # score_and_verify is a non-mutating official/local-gold check.  Keep only
    # candidates that independently re-score at the declared lower cost.
    admitted = []
    for row in payload.get("finalists", []):
        path = ROOT / row["saved_path"]
        check = subprocess.run(
            [sys.executable, str(ROOT / "scripts/golf/verify_candidate_timeout.py"),
             "--task", str(row["task"]), "--onnx", str(path),
             "--timeout", "90", "--label", "restart8018_mid"],
            cwd=ROOT, capture_output=True, text=True,
        )
        output = (check.stdout + check.stderr).strip()
        row["nonmutating_official_gold"] = {
            "returncode": check.returncode,
            "output": output[-4000:],
        }
        try:
            verified = json.loads(check.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            verified = {"ok": False, "reason": "unparseable_verifier_output"}
        row["nonmutating_official_gold"]["result"] = verified
        if (
            check.returncode == 0
            and verified.get("ok") is True
            and verified.get("correct") is True
            and int(verified.get("cost", -1)) == int(row["candidate_cost"])
            and int(row["candidate_cost"]) < int(row["authority_cost"])
        ):
            admitted.append(row)
    payload["finalists"] = admitted
    payload["admission_policy"] = (
        "official/local gold exact; strict checker/static shapes; stable margin; "
        "fresh 2000x2 exact across four ORT configurations; nonmutating verifier"
    )
    payload["root_authority_untouched"] = True
    destination = HERE / f"worker_{args.worker}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "worker": args.worker,
        "assigned": payload["assigned_tasks"],
        "finalists": [
            {"task": row["task"], "cost": row["candidate_cost"],
             "gain": row["score_gain"], "path": row["saved_path"]}
            for row in admitted
        ],
        "elapsed": payload["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
