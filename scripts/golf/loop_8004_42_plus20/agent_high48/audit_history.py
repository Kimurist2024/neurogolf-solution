#!/usr/bin/env python3
"""Screen the complete retained lower frontier and SOUND controls for high48."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (8, 275, 134, 112, 168, 109, 160, 170)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


latest = load_module("high48_latest_impl", HERE.parent / "agent_new_low45" / "audit_latest.py")
known = load_module(
    "high48_known_impl", HERE.parent / "agent_new_low45" / "audit_known_and_rules.py"
)

import sys

sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


ARCHIVES = (
    ROOT / "scripts/golf/loop_7999_13/lane_archive_top200",
    ROOT / "scripts/golf/loop_7999_13/lane_archive_all400",
    ROOT / "scripts/golf/loop_7999_13/lane_archive_loose_sweep",
    ROOT / "scripts/golf/loop_7999_13/lane_archive_zip_sweep",
)
SOUND_CONTROLS = {
    8: [
        ROOT / "scripts/golf/scratch_codex/task008/task008_groundup_v1.onnx",
        ROOT / "artifacts/optimized/task008.onnx",
    ],
    275: [
        ROOT / "scripts/golf/loop_7999_13/lane_a29/task275_shared_gate_router.onnx",
        ROOT / "artifacts/optimized/task275.onnx",
    ],
    134: [
        ROOT / "artifacts/handcrafted/task134.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task134_r06_static322.onnx",
        ROOT / "artifacts/optimized/task134.onnx",
    ],
    112: [
        ROOT / "scripts/golf/loop_7999_13/lane_c5/task112_broadcast_sign_truthful.onnx",
        ROOT / "artifacts/quarantine/task112_7607_cost703_private0_decoded.onnx",
        ROOT / "artifacts/optimized/task112.onnx",
    ],
    168: [
        ROOT / "scripts/golf/loop_7999_13/lane_sound/task168_sound_cost416.onnx",
        ROOT / "scripts/golf/loop_8003_40/agent_sound_local_resume/models/task168.onnx",
    ],
    109: [
        ROOT / "scripts/golf/loop_7999_13/lane_c5/task109_global_l1.onnx",
        ROOT / "artifacts/optimized/task109.onnx",
    ],
    160: [ROOT / "scripts/golf/loop_7999_13/lane_a25/rule_references/task160_truthful_rule_v1.onnx"],
    170: [
        ROOT / "artifacts/quarantine/task170_7503_cost691_private0.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_a15/baseline/task170.onnx",
        ROOT / "artifacts/optimized/task170.onnx",
    ],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actual_cost(path: Path) -> dict[str, int] | dict[str, str]:
    try:
        with tempfile.TemporaryDirectory(prefix="high48_cost_", dir="/tmp"):
            memory, params, cost = cost_of(str(path))
        return {"memory": int(memory), "params": int(params), "cost": int(cost)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect() -> tuple[list[dict[str, object]], dict[str, object]]:
    by_hash: dict[tuple[int, str], dict[str, object]] = {}
    for directory in ARCHIVES:
        for task in TARGETS:
            for path in sorted(directory.glob(f"task{task:03d}_r*.onnx")):
                digest = sha(path)
                key = (task, digest)
                row = by_hash.setdefault(
                    key,
                    {
                        "task": task,
                        "sha256": digest,
                        "path": str(path.relative_to(ROOT)),
                        "sources": [],
                        "kind": "retained_history",
                    },
                )
                row["sources"].append(str(path.relative_to(ROOT)))
    for task, paths in SOUND_CONTROLS.items():
        for path in paths:
            if not path.is_file():
                continue
            digest = sha(path)
            key = (task, digest)
            row = by_hash.setdefault(
                key,
                {
                    "task": task,
                    "sha256": digest,
                    "path": str(path.relative_to(ROOT)),
                    "sources": [],
                    "kind": "sound_control",
                },
            )
            row["sources"].append(str(path.relative_to(ROOT)))
            if row["kind"] == "retained_history":
                row["kind"] = "retained_history_and_sound_control"

    inventory = json.loads((ARCHIVES[1] / "inventory.json").read_text())
    coverage = {
        "source": str((ARCHIVES[1] / "inventory.json").relative_to(ROOT)),
        "stats": inventory["stats"],
        "retained_all400_by_task": {
            str(task): inventory["retained"].get(str(task), []) for task in TARGETS
        },
        "unique_union_models": len(by_hash),
        "unique_union_by_task": {
            str(task): sum(key[0] == task for key in by_hash) for task in TARGETS
        },
    }
    return list(by_hash.values()), coverage


def safe_structure(structure: dict[str, object]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for key in ("checker_full", "strict_shape_data_prop", "static_positive", "standard_domains"):
        if not structure.get(key):
            reasons.append(key)
    if structure.get("banned_ops"):
        reasons.append("banned_ops")
    if structure.get("conv_bias_findings"):
        reasons.append("conv_bias_ub")
    if structure.get("lookup_or_scatter_nodes"):
        reasons.append("lookup_or_scatter")
    if structure.get("giant_einsum"):
        reasons.append("giant_einsum")
    trace = structure.get("runtime_shape_trace")
    if not isinstance(trace, dict) or trace.get("shape_cloak") or trace.get("error"):
        reasons.append("runtime_shape_cloak_or_error")
    return not reasons, reasons


def main() -> None:
    candidates, coverage = collect()
    baselines = json.loads((HERE / "baseline_audit.json").read_text())
    base_costs = {int(row["task"]): int(row["actual_cost"]["cost"]) for row in baselines["targets"]}
    rows: list[dict[str, object]] = []
    for index, seed in enumerate(sorted(candidates, key=lambda row: (int(row["task"]), str(row["sha256"])))):
        task = int(seed["task"])
        path = ROOT / str(seed["path"])
        row = dict(seed)
        row["actual_cost"] = actual_cost(path)
        cost = row["actual_cost"].get("cost") if isinstance(row["actual_cost"], dict) else None
        row["baseline_cost"] = base_costs[task]
        row["strictly_cheaper"] = isinstance(cost, int) and cost < base_costs[task]
        model = onnx.load(path)
        row["structure"] = latest.structural(model, task)
        structure_ok, reasons = safe_structure(row["structure"])
        row["structure_ok"] = structure_ok
        row["structure_reasons"] = reasons
        if row["strictly_cheaper"]:
            row["known_disable_all"] = known.run_known(model, task, True)
            row["known_default"] = known.run_known(model, task, False)
            expected = 265 if task in (168, 160) else 266
            row["known100_dual"] = all(
                isinstance(check, dict)
                and check.get("right") == expected
                and check.get("wrong") == 0
                and check.get("errors") == 0
                for check in (row["known_disable_all"], row["known_default"])
            )
        else:
            row["known_disable_all"] = "NOT_RUN_NOT_STRICTLY_CHEAPER"
            row["known_default"] = "NOT_RUN_NOT_STRICTLY_CHEAPER"
            row["known100_dual"] = False
        row["pre_fresh_finalist"] = bool(
            row["strictly_cheaper"] and row["structure_ok"] and row["known100_dual"]
        )
        if row["strictly_cheaper"] and isinstance(cost, int) and cost > 0:
            row["projected_gain"] = math.log(base_costs[task] / cost)
        else:
            row["projected_gain"] = 0.0
        rows.append(row)
        print(
            f"{index + 1}/{len(candidates)} task{task:03d} cost={cost}/{base_costs[task]} "
            f"struct={structure_ok} known={row['known100_dual']} finalist={row['pre_fresh_finalist']}",
            flush=True,
        )
    output = {
        "coverage": coverage,
        "models_screened": len(rows),
        "strictly_cheaper": sum(bool(row["strictly_cheaper"]) for row in rows),
        "pre_fresh_finalists": [row for row in rows if row["pre_fresh_finalist"]],
        "rows": rows,
    }
    (HERE / "history_audit.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
