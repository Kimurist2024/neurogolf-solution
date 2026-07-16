#!/usr/bin/env python3
"""Independent latest-baseline audit for the eight low40 targets."""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8005.16.zip"
OLD = ROOT / "submission_base_7999.13.zip"
TARGETS = (22, 181, 104, 294, 128, 152, 203, 236)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
CLOAK_OR_LOOKUP = {
    "CenterCropPad", "ConstantOfShape", "GatherND", "GroupNormalization",
    "ReduceLogSum", "Resize", "ScatterElements", "Shrink",
}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402
from scripts.golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_private_exact15.audit_exact import trace_shapes  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


def run_known(model: onnx.ModelProto, task: int, disable: bool) -> dict[str, object]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    examples = scoring.load_examples(task)
    converted = []
    skipped = 0
    for subset in ("train", "test", "arc-gen"):
        for example in examples.get(subset, []):
            item = scoring.convert_to_numpy(example)
            if item is None:
                skipped += 1
            else:
                converted.append(item)
    try:
        session = ort.InferenceSession(sanitized.SerializeToString(), options)
    except Exception as exc:
        return {
            "right": 0, "wrong": 0, "errors": len(converted), "skipped": skipped,
            "total": len(converted), "runtime_output_shapes": [],
            "session_error": f"{type(exc).__name__}: {exc}",
        }
    right = wrong = errors = 0
    shapes: set[tuple[int, ...]] = set()
    for item in converted:
        try:
            raw = session.run(["output"], {"input": item["input"]})[0]
            shapes.add(tuple(int(value) for value in raw.shape))
            if np.array_equal(raw > 0, item["output"] > 0):
                right += 1
            else:
                wrong += 1
        except Exception:
            errors += 1
    return {
        "right": right, "wrong": wrong, "errors": errors, "skipped": skipped,
        "total": len(converted),
        "runtime_output_shapes": [list(shape) for shape in sorted(shapes)],
    }


def audit_task(task: int) -> dict[str, object]:
    path = HERE / f"base/task{task:03d}.onnx"
    model = onnx.load(path)
    checker = strict = True
    checker_error = strict_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        checker = False
        checker_error = f"{type(exc).__name__}: {exc}"
    try:
        shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    except Exception as exc:
        strict = False
        strict_error = f"{type(exc).__name__}: {exc}"
    disabled = run_known(model, task, True)
    default = run_known(model, task, False)
    try:
        trace = trace_shapes(model, task)
    except Exception as exc:
        trace = {"shape_cloak": True, "error": f"{type(exc).__name__}: {exc}"}
    memory, params, cost = cost_of(str(path))
    ops = collections.Counter(node.op_type for node in model.graph.node)
    max_einsum = max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)
    domains = sorted(
        {item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")}
        | {node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")}
    )
    declared_output = dims(model.graph.output[0])
    observed = disabled.get("runtime_output_shapes", [])
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path.read_bytes()),
        "file_bytes": path.stat().st_size,
        "actual_cost": {"memory": memory, "params": params, "cost": cost},
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "op_histogram": dict(sorted(ops.items())),
        "max_einsum_inputs": max_einsum,
        "giant_einsum": max_einsum > 16,
        "cloak_or_lookup_ops": sorted(set(ops) & CLOAK_OR_LOOKUP),
        "checker_full": checker,
        "checker_error": checker_error,
        "strict_data_prop": strict,
        "strict_error": strict_error,
        "standard_domains": not domains,
        "nonstandard_domains": domains,
        "banned_ops": sorted({node.op_type for node in model.graph.node if node.op_type in BANNED or "Sequence" in node.op_type}),
        "conv_bias_findings": check_conv_bias(model),
        "declared_output_shape": declared_output,
        "runtime_output_shape_truthful": len(observed) == 1 and observed[0] == declared_output,
        "runtime_shape_trace": trace,
        "known_disable_all": disabled,
        "known_default": default,
    }


def main() -> None:
    base_sha = sha(BASE.read_bytes())
    with zipfile.ZipFile(BASE) as latest, zipfile.ZipFile(OLD) as old:
        same = {
            str(task): latest.read(f"task{task:03d}.onnx") == old.read(f"task{task:03d}.onnx")
            for task in TARGETS
        }
    rows = {str(task): audit_task(task) for task in TARGETS}
    baseline = {
        "baseline": BASE.name,
        "baseline_sha256": base_sha,
        "targets": list(TARGETS),
        "same_payload_as_7999_13": same,
        "rows": rows,
    }
    (HERE / "baseline_audit.json").write_text(json.dumps(baseline, indent=2) + "\n")

    harvest = json.loads((ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json").read_text())
    archive = json.loads((ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text())
    exact = json.loads((ROOT / "scripts/golf/loop_8004_42_plus20/agent_exact_wave2/static_scan.json").read_text())
    einsum = json.loads((ROOT / "scripts/golf/loop_8004_42_plus20/root_exact_einsum25/scan_report.json").read_text())
    exact_opportunities = [
        {"family": family, **row}
        for family, family_rows in exact.get("opportunities", {}).items()
        for row in family_rows
        if row.get("task") in TARGETS
    ]
    einsum_candidates = [
        {"family": family, **row}
        for family in ("initializer_dedup", "outer_fusion", "sign_absorption")
        for row in einsum.get(family, [])
        if row.get("task") in TARGETS
    ]
    history = {
        "latest_baseline": {"path": BASE.name, "sha256": base_sha},
        "all_members_unchanged_from_7999_13": all(same.values()),
        "harvest_inventory": harvest["inventory"],
        "harvest_target_rows": {
            str(task): [row for row in harvest["rows"] if row.get("task") == task]
            for task in TARGETS
        },
        "all400_archive_inventory": {
            "baseline": archive["base"],
            "stats": archive["stats"],
            "retained_below_baseline": {str(task): archive["retained"].get(str(task), []) for task in TARGETS},
        },
        "exact_wave2": {
            "tasks_scanned": exact["tasks_scanned"],
            "accepted": exact["summary"].get("accepted", 0),
            "target_opportunities": exact_opportunities,
            "target_candidates": [row for row in exact.get("candidates", []) if row.get("task") in TARGETS],
        },
        "exact_einsum_all400": {
            "task_count": einsum.get("task_count"),
            "target_candidates": einsum_candidates,
        },
    }
    (HERE / "history_audit.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
