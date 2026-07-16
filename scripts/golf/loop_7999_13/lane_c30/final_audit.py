#!/usr/bin/env python3
"""Final C30 evidence: dual-runtime incumbents plus all cheaper rejections."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


def load_shared() -> Any:
    path = HERE.parent / "lane_b15" / "audit_candidates.py"
    spec = importlib.util.spec_from_file_location("c30_shared_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static_positive(value: onnx.ValueInfoProto) -> bool:
    return all(
        dimension.HasField("dim_value") and dimension.dim_value > 0
        for dimension in value.type.tensor_type.shape.dim
    )


def structure(model: onnx.ModelProto) -> dict[str, Any]:
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    einsums = [node for node in model.graph.node if node.op_type == "Einsum"]
    return {
        "checker_full": True,
        "strict_data_prop": True,
        "canonical_io": (
            len(model.graph.input) == 1
            and len(model.graph.output) == 1
            and model.graph.input[0].name == "input"
            and model.graph.output[0].name == "output"
        ),
        "static_positive_shapes": all(static_positive(value) for value in values),
        "no_external_initializers": all(
            item.data_location != onnx.TensorProto.EXTERNAL and not item.external_data
            for item in model.graph.initializer
        ),
        "finite_initializers": all(
            array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
            for array in (numpy_helper.to_array(item) for item in model.graph.initializer)
        ),
        "conv_ub_findings": check_conv_bias(model),
        "existing_giant_einsum_count": len(einsums),
        "existing_giant_einsum_operands": [len(node.input) for node in einsums],
        "giant_policy": "grandfathered incumbent only; no candidate enlarged it",
    }


def main() -> int:
    shared = load_shared()
    incumbents = []
    for task in (50, 287):
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        model = onnx.load(path)
        score = cost_of(path)
        row = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "cost": {"memory": score[0], "params": score[1], "total": score[2]},
            "structure": structure(model),
            "runtime_shapes": shared.trace_runtime_shapes(copy.deepcopy(model), task),
            "known_dual": shared.known_dual(copy.deepcopy(model), task),
        }
        incumbents.append(row)
        print(task, row["cost"], row["known_dual"], flush=True)

    probe_screen = json.loads((HERE / "task050_probe_screen.json").read_text())
    gather_external = json.loads((HERE / "task287_gather_external.json").read_text())
    report = {
        "assignment": [50, 287],
        "incumbents": incumbents,
        "cheaper_rejections": {
            "task050": {
                "candidates": probe_screen,
                "cost_each": 84,
                "baseline_cost": 88,
                "reason": "all four fail train[0] in both ORT modes",
                "fresh5000": "not run because known gate failed",
            },
            "task287": {
                "candidate": "others/1/62901/task287_cost30_gather (1).onnx",
                "candidate_cost": 30,
                "baseline_cost": 74,
                "external_known": gather_external["candidate"]["known"],
                "reason": "wrong on all four train examples and fixed-index lookup is prohibited",
                "fresh5000": "not run because known gate failed",
            },
        },
        "winners": [],
        "projected_gain": 0.0,
        "conclusion": "No eligible lower-cost model; retain both exact incumbents.",
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(
        json.dumps({"winners": [], "projected_gain": 0.0}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
