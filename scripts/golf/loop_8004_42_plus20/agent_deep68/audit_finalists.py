#!/usr/bin/env python3
"""Final fail-closed screen for the four deep68 closest candidates."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import (  # noqa: E402
    run_known,
    structure,
)


CANDIDATES = {
    285: HERE / "candidates/task285_true_dedup.onnx",
    366: HERE / "candidates/task366_truthful_history_dedup.onnx",
    286: ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task286_r01_static7122.onnx",
    233: ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task233_r01_static4936.onnx",
}

FRESH = {
    285: {"not_run": "candidate already fails strict-lower pre-gate"},
    366: {
        "source_computation_historical": {"right": 4685, "wrong": 72, "executable": 4757, "accuracy": 4685 / 4757},
        "source": "scripts/golf/loop_7999_13/lane_c17/fresh_evidence.json",
        "note": "truthful repair has identical computation but already fails strict-lower pre-gate",
    },
    286: {
        "dual_mode": {"right": 4318, "wrong": 682, "total": 5000, "accuracy": 4318 / 5000},
        "strict": {"right": 4294, "wrong": 706, "total": 5000, "accuracy": 4294 / 5000},
        "sources": [
            "scripts/golf/loop_7999_13/lane_root21_task286_dual5000.json",
            "scripts/golf/loop_7999_13/lane_root21_task286_strict5000.json",
        ],
    },
    233: {
        "right": 0,
        "wrong": 100,
        "total": 100,
        "accuracy": 0.0,
        "source": "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task233_268_k100.json",
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = []
    with zipfile.ZipFile(ROOT / "submission_base_8005.16.zip") as archive:
        for task, path in CANDIDATES.items():
            baseline_path = HERE / f"baselines/task{task:03d}.onnx"
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_bytes(archive.read(f"task{task:03d}.onnx"))
            baseline = cost_of(str(baseline_path))
            candidate = cost_of(str(path))
            model = onnx.load(path)
            structural = structure(copy.deepcopy(model), task)
            disable = run_known(copy.deepcopy(model), task, True)
            default = run_known(copy.deepcopy(model), task, False)
            runtime = structural.get("runtime_shapes") or {}
            strict_lower = candidate[2] < baseline[2]
            known_dual = (
                disable.get("right") == disable.get("total")
                and disable.get("errors") == 0
                and default.get("right") == default.get("total")
                and default.get("errors") == 0
            )
            truthful = not runtime.get("shape_cloak", True) and not runtime.get("error")
            lookup = structural.get("lookup_or_scatter") or []
            has_lookup_abuse = any(op in {"TfIdfVectorizer", "Hardmax"} for op in lookup)
            reasons = []
            if not strict_lower:
                reasons.append("not_strictly_lower_than_immutable_8005_16_member")
            if not known_dual:
                reasons.append("known_dual_not_perfect")
            if not truthful:
                reasons.append("runtime_shapes_not_truthful")
            if has_lookup_abuse:
                reasons.append("lookup_abuse")
            if task == 286:
                reasons.append("fresh_accuracy_below_90_percent")
            if task == 233:
                reasons.append("fresh_accuracy_zero")
            if task == 366:
                reasons.append("historical_computation_not_exact_true_rule")
            rows.append(
                {
                    "task": task,
                    "baseline": {"memory": baseline[0], "params": baseline[1], "cost": baseline[2]},
                    "candidate": {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha(path),
                        "memory": candidate[0],
                        "params": candidate[1],
                        "cost": candidate[2],
                    },
                    "strict_lower": strict_lower,
                    "known_disable_all": disable,
                    "known_default": default,
                    "known_dual_perfect": known_dual,
                    "structure": structural,
                    "truthful_runtime_shapes": truthful,
                    "fresh": FRESH[task],
                    "verdict": "SAFE_WINNER" if not reasons else "REJECT",
                    "reasons": reasons,
                }
            )
    result = {
        "baseline_zip": "submission_base_8005.16.zip",
        "baseline_zip_sha256": sha(ROOT / "submission_base_8005.16.zip"),
        "rows": rows,
        "safe_winners": [row for row in rows if row["verdict"] == "SAFE_WINNER"],
    }
    (HERE / "finalist_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"safe_winners": len(result["safe_winners"]), "tasks": {row["task"]: row["reasons"] for row in rows}}, indent=2))


if __name__ == "__main__":
    main()
