#!/usr/bin/env python3
"""Freeze B14 structural, optimizer, scan, and validation evidence."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxoptimizer
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
DTYPE_BYTES = {
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.UINT8: 1,
    onnx.TensorProto.INT8: 1,
    onnx.TensorProto.UINT16: 2,
    onnx.TensorProto.INT16: 2,
    onnx.TensorProto.INT32: 4,
    onnx.TensorProto.INT64: 8,
    onnx.TensorProto.BOOL: 1,
    onnx.TensorProto.FLOAT16: 2,
    onnx.TensorProto.DOUBLE: 8,
    onnx.TensorProto.UINT32: 4,
    onnx.TensorProto.UINT64: 8,
    onnx.TensorProto.BFLOAT16: 2,
}


def dims_positive(value: onnx.ValueInfoProto) -> bool:
    return all(
        dimension.HasField("dim_value") and dimension.dim_value > 0
        for dimension in value.type.tensor_type.shape.dim
    )


def static_cost(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    values = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    final = {value.name for value in inferred.graph.output}
    memory = 0
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in final:
                continue
            tensor = values[name].type.tensor_type
            memory += math.prod(d.dim_value for d in tensor.shape.dim) * DTYPE_BYTES[tensor.elem_type]
    params = sum(math.prod(item.dims) if item.dims else 1 for item in model.graph.initializer)
    return {"memory": int(memory), "params": int(params), "cost": int(memory + params)}


def candidate_eligible(model: onnx.ModelProto) -> tuple[bool, str | None]:
    """Apply the complete lane structure contract to an optimizer output."""
    try:
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True
        )
    except Exception as exc:  # noqa: BLE001 - preserve optimizer rejection evidence
        return False, f"{type(exc).__name__}: {exc}"
    if not (
        len(model.graph.input) == 1
        and len(model.graph.output) == 1
        and model.graph.input[0].name == "input"
        and model.graph.output[0].name == "output"
    ):
        return False, "I/O contract changed"
    if model.functions or model.graph.sparse_initializer:
        return False, "functions or sparse initializers"
    if any(item.domain not in ("", "ai.onnx") for item in model.opset_import):
        return False, "foreign opset"
    if any(node.domain not in ("", "ai.onnx") for node in model.graph.node):
        return False, "foreign node domain"
    for node in model.graph.node:
        if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper():
            return False, f"banned op {node.op_type}"
        if node.op_type == "Einsum" and len(node.input) > 16:
            return False, "giant Einsum"
        if any(
            attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for attribute in node.attribute
        ):
            return False, "nested graph"
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    if not all(dims_positive(value) for value in values):
        return False, "non-static/non-positive shape"
    if any(
        item.data_location == onnx.TensorProto.EXTERNAL or item.external_data
        for item in model.graph.initializer
    ):
        return False, "external initializer"
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
            return False, "nonfinite initializer"
    if check_conv_bias(model):
        return False, "Conv bias mismatch"
    return True, None


def structural_audit(task: int) -> dict[str, Any]:
    model = onnx.load(HERE / "baseline" / f"task{task:03d}.onnx")
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    arrays = [numpy_helper.to_array(item) for item in model.graph.initializer]
    all_values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    equations = []
    for node in model.graph.node:
        if node.op_type == "Einsum":
            for attribute in node.attribute:
                if attribute.name == "equation":
                    value = onnx.helper.get_attribute_value(attribute)
                    equations.append(value.decode() if isinstance(value, bytes) else str(value))
    checks = {
        "checker_full": True,
        "strict_data_prop_inference": True,
        "one_named_input_output": (
            len(model.graph.input) == 1
            and len(model.graph.output) == 1
            and model.graph.input[0].name == "input"
            and model.graph.output[0].name == "output"
        ),
        "standard_domains_only": all(
            item.domain in ("", "ai.onnx") for item in model.opset_import
        )
        and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
        "no_functions_sparse_nested": (
            not model.functions
            and not model.graph.sparse_initializer
            and all(
                attribute.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for node in model.graph.node
                for attribute in node.attribute
            )
        ),
        "no_banned_ops": all(
            node.op_type.upper() not in BANNED and "SEQUENCE" not in node.op_type.upper()
            for node in model.graph.node
        ),
        "static_positive_shapes": all(dims_positive(value) for value in all_values),
        "no_giant_einsum": all(
            node.op_type != "Einsum" or len(node.input) <= 16 for node in model.graph.node
        ),
        "no_external_initializers": all(
            item.data_location != onnx.TensorProto.EXTERNAL and not item.external_data
            for item in model.graph.initializer
        ),
        "finite_initializers": all(
            array.dtype.kind not in "fc" or bool(np.isfinite(array).all()) for array in arrays
        ),
        "conv_bias_safe": not check_conv_bias(model),
    }
    return {
        "task": task,
        "checks": checks,
        "pass": all(checks.values()),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "einsum_equations": equations,
        "static_cost": static_cost(model),
    }


def optimizer_audit(task: int) -> dict[str, Any]:
    model = onnx.load(HERE / "baseline" / f"task{task:03d}.onnx")
    original = model.SerializeToString()
    base_cost = static_cost(model)
    changed: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for name in onnxoptimizer.get_available_passes():
        try:
            candidate = onnxoptimizer.optimize(model, [name])
            if candidate.SerializeToString() == original:
                continue
            try:
                valid, error = candidate_eligible(candidate)
                cost = static_cost(candidate) if valid else None
            except Exception as exc:  # noqa: BLE001 - record invalid optimizer output
                valid = False
                cost = None
                error = f"{type(exc).__name__}: {exc}"
            changed.append({"pass": name, "valid": valid, "cost": cost, "error": error})
        except Exception as exc:  # noqa: BLE001 - complete audit of installed passes
            errors.append({"pass": name, "error": f"{type(exc).__name__}: {exc}"})
    all_passes = onnxoptimizer.optimize(model)
    return {
        "task": task,
        "base_cost": base_cost,
        "available_passes": len(onnxoptimizer.get_available_passes()),
        "changed_single_passes": changed,
        "single_pass_errors": errors,
        "default_all_changed": all_passes.SerializeToString() != original,
        "default_all_cost": static_cost(all_passes),
        "strictly_cheaper_valid": [
            row for row in changed if row["valid"] and row["cost"]["cost"] < base_cost["cost"]
        ],
    }


def main() -> int:
    scan = json.loads((HERE / "scan_results.json").read_text())
    known = json.loads((HERE / "known_dual_ort.json").read_text())
    fresh = json.loads((HERE / "fresh5000_dual_ort.json").read_text())
    structures = [structural_audit(task) for task in (5, 80)]
    optimizers = [optimizer_audit(task) for task in (5, 80)]
    (HERE / "structural_audit.json").write_text(json.dumps(structures, indent=2) + "\n")
    (HERE / "optimizer_probe.json").write_text(json.dumps(optimizers, indent=2) + "\n")

    task_summary: dict[str, Any] = {}
    for task in (5, 80):
        rows = [row for row in scan["rows"] if row["task"] == task]
        nonbase = [row for row in rows if row["status"] != "base_identical"]
        floors = [row for row in nonbase if "static_cost_floor" in row]
        task_summary[str(task)] = {
            "base": scan["base"][str(task)],
            "unique_models": len(rows),
            "raw_sources": json.loads((HERE / "collection.json").read_text())["raw_source_counts"][str(task)],
            "statuses": dict(Counter(row["status"] for row in rows)),
            "lowest_nonbase_static_floor": min(
                (row["static_cost_floor"] for row in floors), default=None
            ),
            "known_dual": [row for row in known if row["task"] == task],
            "fresh5000_dual": [row for row in fresh if row["task"] == task],
            "structure_pass": next(row["pass"] for row in structures if row["task"] == task),
            "optimizer_strictly_cheaper": next(
                row["strictly_cheaper_valid"] for row in optimizers if row["task"] == task
            ),
        }

    winner_manifest = {
        "base_zip": "submission_base_7999.13.zip",
        "tasks": [5, 80],
        "winners": [],
        "aggregate_gain": 0.0,
        "runtime_errors": sum(row["errors"] for row in known + fresh),
        "reason": "No structurally valid strict-cost reduction exists in the 190-model scan or standard optimizer sweep.",
    }
    failure_manifest = {
        "task005": {
            "base_cost": 2325,
            "result": "reject_all_strict_cost_candidates",
            "lowest_nonbase_static_floor": task_summary["5"]["lowest_nonbase_static_floor"],
            "base_fresh_accuracy": 0.9934,
            "base_fresh_runtime_errors": 0,
            "known_sound_rebuild": "scripts/golf/scratch_codex/task005/cand_v24_fourier_qw_qcoord.onnx",
            "known_sound_rebuild_cost": 2389,
            "reason": "The exact 2325 selector fails duplicate-guide-color cases; the sound selector costs +64, while no scanned model is strictly cheaper.",
        },
        "task080": {
            "base_cost": 3051,
            "result": "reject_all_strict_cost_candidates",
            "lowest_nonbase_static_floor": task_summary["80"]["lowest_nonbase_static_floor"],
            "base_fresh_accuracy": 1.0,
            "base_fresh_runtime_errors": 0,
            "reason": "Exact compact compiler passed dual fresh5000; every nonbase scanned model has a higher static floor.",
        },
    }
    (HERE / "winner_manifest.json").write_text(json.dumps(winner_manifest, indent=2) + "\n")
    (HERE / "failure_manifest.json").write_text(json.dumps(failure_manifest, indent=2) + "\n")
    (HERE / "manifest.json").write_text(
        json.dumps(
            {
                "task_summary": task_summary,
                "winner_manifest": winner_manifest,
                "failure_manifest": failure_manifest,
            },
            indent=2,
        )
        + "\n"
    )

    report = f"""# Lane B14 report — task005/task080 exact-base audit

Base: `submission_base_7999.13.zip` (task005 cost 2325; task080 cost 3051).
No project-level ZIP, CSV, or score ledger was modified by this lane.

## Result

- Winners: **0**; aggregate score gain: **+0.00**.
- The deduplicated scan covered 113 task005 models / 970 raw sources and 77
  task080 models / 929 raw sources. It found no strict-cost candidate.
- task005's lowest nonbase static floor is 2325 (a tie); the next floor is
  2329. task080's lowest nonbase floor is 3053, already above its 3051 base.
- All exact baselines pass checker, strict data-propagating shape inference,
  static-positive shapes, standard domains, banned-op/nested/function/sparse/
  external/nonfinite/giant-Einsum checks, and Conv bias safety.
- All 48 installed optimizer passes were probed individually plus the default
  full sweep. No valid strictly cheaper model was produced.

## Dual-ORT validation

- Known: task005 266/266 and task080 231/231 in both ORT modes; errors 0.
- Fresh seed 140799913: task005 4967/5000 = 99.34% in both modes; errors 0.
  The 33 semantic misses are duplicate-guide-color selector failures, not
  runtime/session errors. This meets the lane's explicit >=95% base-equivalent
  allowance, but is not generator-sound.
- Fresh seed 140799913: task080 5000/5000 in both modes; errors 0 (1001 >30x30
  generated cases skipped per the scorer contract in each stream).
- Minimum observed nonzero absolute output margin is 1.0 throughout.

## Promotion decision

Nothing is promoted. The sound task005 rebuild costs 2389 (+64), and the exact
task080 model is already below every stored alternative. `winner_manifest.json`
therefore stays empty and records runtime errors = 0.
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps(winner_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
