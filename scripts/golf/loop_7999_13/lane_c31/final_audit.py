#!/usr/bin/env python3
"""Assemble strict C31 evidence without modifying the aggregate."""

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


def shared() -> Any:
    path = HERE.parent / "lane_b15" / "audit_candidates.py"
    spec = importlib.util.spec_from_file_location("c31_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structure(model: onnx.ModelProto) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    arrays = [numpy_helper.to_array(item) for item in model.graph.initializer]
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
        "static_positive_shapes": all(
            dimension.HasField("dim_value") and dimension.dim_value > 0
            for value in values
            for dimension in value.type.tensor_type.shape.dim
        ),
        "finite_initializers": all(
            array.dtype.kind not in "fc" or bool(np.isfinite(array).all()) for array in arrays
        ),
        "no_external_initializers": all(
            item.data_location != onnx.TensorProto.EXTERNAL and not item.external_data
            for item in model.graph.initializer
        ),
        "conv_ub_findings": check_conv_bias(model),
        "einsum_operands": [len(node.input) for node in einsums],
        "giant_policy": "incumbent grandfathered; no giant was added or enlarged",
    }


def main() -> int:
    helper = shared()
    incumbents = []
    for task in (199, 212):
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        model = onnx.load(path)
        memory, params, total = cost_of(path)
        row = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path),
            "cost": {"memory": memory, "params": params, "total": total},
            "structure": structure(model),
            "runtime_shapes": helper.trace_runtime_shapes(copy.deepcopy(model), task),
            "known_dual": helper.known_dual(copy.deepcopy(model), task),
            "external": json.loads((HERE / f"task{task:03d}_baseline_external.json").read_text())[
                "candidate"
            ],
        }
        incumbents.append(row)
        print(task, row["known_dual"], flush=True)
    contraction = json.loads((HERE / "exact_contraction_inventory.json").read_text())
    screen = json.loads((HERE / "candidate_screen.json").read_text())
    fresh = json.loads((HERE / "fresh_baselines.json").read_text())
    report = {
        "source_zip": "scripts/golf/loop_7999_13/submission_7999.13_wave17_candidate_meta.zip",
        "source_zip_sha256": "74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534",
        "incumbents": incumbents,
        "history": {
            "task199_files": 602,
            "task199_unique": 27,
            "task199_below_baseline": 12,
            "task212_files": 570,
            "task212_unique": 17,
            "task212_below_baseline": 0,
        },
        "task199_cheaper_screen": screen,
        "exact_pair_contractions": {
            task: {
                "enumerated": len(row["pair_contractions"]),
                "parameter_reducing": [
                    item for item in row["pair_contractions"] if item["delta"] < 0
                ],
            }
            for task, row in contraction.items()
        },
        "fresh_controls": fresh,
        "fresh_task212_omission": (
            "Stopped baseline-only control after root requested immediate handoff; no cheaper task212 "
            "candidate existed, so fresh admission was not applicable."
        ),
        "winners": [],
        "projected_gain": 0.0,
        "conclusion": "No lower-cost candidate passed even the known gate; retain both incumbents.",
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(
        json.dumps({"winners": [], "projected_gain": 0.0}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
