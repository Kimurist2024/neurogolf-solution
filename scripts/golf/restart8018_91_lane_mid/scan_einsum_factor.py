#!/usr/bin/env python3
"""Scoped exact Einsum outer-factor/dedup scan on the 8018.91 authority."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8018.91.zip"
AUTHORITY_SHA256 = "e43865760ec8807fbb217fba718226ca6b86d9128b911479214e3252b9f9e091"
EXCLUDED = {
    9, 12, 15, 23, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 96, 101,
    102, 110, 112, 118, 133, 134, 138, 145, 157, 158, 161, 168, 169,
    170, 173, 174, 175, 178, 182, 185, 187, 188, 191, 192, 196, 198,
    202, 204, 205, 208, 209, 216, 219, 222, 233, 246, 251, 255, 273,
    277, 285, 286, 302, 319, 325, 333, 343, 346, 354, 355, 361, 365,
    366, 372, 377, 379, 391, 393, 396,
}


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact(row: dict[str, Any]) -> bool:
    return bool(
        row.get("right") == row.get("total")
        and row.get("wrong") == 0 and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("early_reject_reason") is None
    )


def main() -> int:
    if hashlib.sha256(AUTHORITY.read_bytes()).hexdigest() != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA drift")
    outer = import_path(
        "restart8018_mid_outer",
        ROOT / "scripts/golf/root_einsum_outer_factor_scan_270/scan_einsum_outer_factor.py",
    )
    base = import_path(
        "restart8018_mid_factor_base",
        ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py",
    )
    base.THRESHOLD = 1.0
    base.FRESH_PER_SEED = 2_000
    base.SUPPORT.POLICY_THRESHOLD = 1.0
    base.SUPPORT.FRESH_PER_SEED = 2_000
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        tasks = [int(row["task"].removeprefix("task")) for row in csv.DictReader(handle)
                 if 250 <= int(row["cost"]) <= 399
                 and int(row["task"].removeprefix("task")) not in EXCLUDED]
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    report: dict[str, Any] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "method": "exact serialized outer-product factorization with cross-initializer dedup DP",
        "tasks": [], "finalists": [],
    }
    candidate_dir = HERE / "candidates_factor"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in tasks:
            data = archive.read(f"task{task:03d}.onnx")
            _model, arrays, groups, discovery = outer.discover_internal(
                task, "authority8018_91", data
            )
            best, stats = outer.optimize_groups(arrays, groups)
            optimization = outer.public_optimization(best, stats)
            row: dict[str, Any] = {
                "task": task, "discovery": discovery,
                "optimization": optimization,
            }
            report["tasks"].append(row)
            if best is None or int(best["projected_param_saving"]) <= 0:
                continue
            candidate, build = outer.build_candidate(data, best)
            row["build"] = build
            digest = hashlib.sha256(candidate).hexdigest()
            model = onnx.load_model_from_string(candidate)
            reasons = base.quick_preflight(model)
            row["preflight_reasons"] = reasons
            if reasons:
                continue
            cases, counts = base.SUPPORT.known_cases(task)
            authority_profile = base.POLICY.fast_profile(
                base.SUPPORT, task, onnx.load_model_from_string(data), cases[0]
            )
            profile = base.POLICY.fast_profile(base.SUPPORT, task, model, cases[0])
            row["known_counts"] = counts
            row["authority_profile"] = authority_profile
            row["candidate_profile"] = profile
            if (authority_profile is None or profile is None
                    or int(profile["cost"]) >= int(authority_profile["cost"])):
                continue
            known = base.SUPPORT.evaluate_four(candidate, cases)
            row["known_four"] = {name: base.compact_runtime(value)
                                 for name, value in known.items()}
            if not all(exact(value) for value in known.values()):
                continue
            structure = base.POLICY.structure_audit(base.SUPPORT, task, model, candidate)
            row["structure"] = structure
            if not structure["pass"]:
                continue
            fresh = []
            for seed in (818_930_000 + task, 818_940_000 + task):
                fresh_cases, generation = base.SUPPORT.fresh_cases(task, seed, task_map)
                runtime = base.SUPPORT.evaluate_four(candidate, fresh_cases)
                fresh.append({
                    "seed": seed, "generation": generation, "count": len(fresh_cases),
                    "runtime": {name: base.compact_runtime(value)
                                for name, value in runtime.items()},
                    "pass": len(fresh_cases) >= 2_000
                    and all(exact(value) for value in runtime.values()),
                })
            row["fresh"] = fresh
            if not all(run["pass"] for run in fresh):
                continue
            path = candidate_dir / (
                f"task{task:03d}_outer_factor_cost{profile['cost']}_{digest[:12]}.onnx"
            )
            path.write_bytes(candidate)
            check = subprocess.run(
                [sys.executable, str(ROOT / "scripts/golf/verify_candidate_timeout.py"),
                 "--task", str(task), "--onnx", str(path), "--timeout", "90",
                 "--label", "restart8018_factor"],
                cwd=ROOT, capture_output=True, text=True,
            )
            try:
                verified = json.loads(check.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                verified = {"ok": False, "reason": "unparseable"}
            row["nonmutating_official_gold"] = {
                "returncode": check.returncode, "result": verified,
                "output": (check.stdout + check.stderr)[-4000:],
            }
            if not (check.returncode == 0 and verified.get("ok") is True
                    and verified.get("correct") is True
                    and int(verified.get("cost", -1)) == int(profile["cost"])):
                continue
            winner = {
                "task": task,
                "authority_cost": int(authority_profile["cost"]),
                "candidate_cost": int(profile["cost"]),
                "gain": math.log(int(authority_profile["cost"]) / int(profile["cost"])),
                "sha256": digest,
                "path": str(path.relative_to(ROOT)),
                "evidence": row,
            }
            report["finalists"].append(winner)
            print(json.dumps({"STRICT_WINNER": task, "cost": profile["cost"],
                              "gain": winner["gain"], "path": winner["path"]}), flush=True)
    (HERE / "einsum_factor_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"tasks": len(report["tasks"]),
                      "factorable": sum(bool(r["discovery"].get("factorable_initializers"))
                                        for r in report["tasks"]),
                      "finalists": len(report["finalists"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
