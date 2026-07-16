#!/usr/bin/env python3
"""Rank LB-probe candidates on two independent fresh-500 seeds."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (
    75, 392, 225, 218, 159, 185, 263, 370, 182, 330,
    361, 157, 280, 382, 201, 251, 12, 107, 131, 364,
)
BASE_ZIP = ROOT / "submission_base_8006.61.zip"
COSTS_PATH = HERE / "baseline_costs_8006_61.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCANNER = load_module(
    "expand20f_fresh_scanner",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py",
)


def main() -> int:
    reopened = json.loads((HERE / "audit" / "reopened_giant_lookup_private.json").read_text())
    probes = [row for row in reopened["rows"] if row["classification"] == "LB_PROBE_REQUIRED"]
    wanted = {row["sha256"] for row in probes}
    SCANNER.HERE = HERE
    SCANNER.TARGETS = TARGETS
    SCANNER.BASE_ZIP = BASE_ZIP
    SCANNER.CURRENT_COSTS_JSON = COSTS_PATH
    inventory, _ = SCANNER.inventory()
    data_by_sha = {
        digest: item["data"]
        for per_task in inventory.values()
        for digest, item in per_task.items()
        if digest in wanted
    }
    candidate_dir = HERE / "candidates" / "lb_probe_required"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    manifest = []
    for task in sorted({int(row["task"]) for row in probes}):
        task_rows = []
        for row in probes:
            if int(row["task"]) != task:
                continue
            data = data_by_sha[row["sha256"]]
            path = candidate_dir / f"task{task:03d}_{row['sha256'][:12]}_cost{row['profile']['cost']}.onnx"
            path.write_bytes(data)
            row["isolated_candidate"] = str(path.relative_to(ROOT))
            task_rows.append({**row, "data": data})
        seeds = (80_066_100 + task, 80_166_100 + task)
        seed_reports = [SCANNER.fresh_dual(task, task_rows, 500, seed) for seed in seeds]
        reports.append({"task": task, "seeds": list(seeds), "reports": seed_reports})
        for row in task_rows:
            digest = row["sha256"]
            outcomes = []
            for seed_report in seed_reports:
                per_mode = seed_report["candidates"][digest]
                outcomes.append(
                    {
                        "seed": seed_report["seed"],
                        "modes": per_mode,
                        "minimum_rate": min(item["right"] / 500 for item in per_mode.values()),
                        "fresh100": all(
                            item["right"] == 500
                            and item["wrong"] == 0
                            and item["errors"] == 0
                            and not item.get("session_error")
                            for item in per_mode.values()
                        ),
                    }
                )
            fresh100_both = all(outcome["fresh100"] for outcome in outcomes)
            manifest.append(
                {
                    "task": task,
                    "sha256": digest,
                    "baseline_cost": row["baseline_cost"],
                    "actual_cost": row["profile"]["cost"],
                    "isolated_candidate": row["isolated_candidate"],
                    "known_x4_complete": True,
                    "truthful_runtime_shapes": True,
                    "fresh": outcomes,
                    "fresh100_both_seeds": fresh100_both,
                    "classification": "LB_PROBE_REQUIRED",
                    "fixed_safe": False,
                    "fixed_safe_blocker": "private/lookup lineage lacks LB-white or all-input equivalence proof",
                }
            )
    output = {
        "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": reopened["baseline_zip_sha256"],
        "fresh_count_per_seed": 500,
        "candidate_count": len(manifest),
        "fresh100_both_seed_count": sum(row["fresh100_both_seeds"] for row in manifest),
        "reports": reports,
    }
    (HERE / "audit" / "fresh_probe_2seed.json").write_text(json.dumps(output, indent=2) + "\n")
    (HERE / "probe_manifest.json").write_text(
        json.dumps(
            {
                "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
                "baseline_zip_sha256": reopened["baseline_zip_sha256"],
                "classification": "LB_PROBE_REQUIRED",
                "candidates": sorted(manifest, key=lambda row: (row["actual_cost"], row["sha256"])),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
