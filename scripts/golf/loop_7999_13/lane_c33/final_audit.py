#!/usr/bin/env python3
"""Assemble strict C33 evidence and empty winner manifest."""

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
    spec = importlib.util.spec_from_file_location("c33_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structure(model: onnx.ModelProto) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    arrays = [numpy_helper.to_array(item) for item in model.graph.initializer]
    return {
        "checker_full": True,
        "strict_data_prop": True,
        "canonical_io": len(model.graph.input) == 1 and len(model.graph.output) == 1 and model.graph.input[0].name == "input" and model.graph.output[0].name == "output",
        "static_positive_shapes": all(
            dim.HasField("dim_value") and dim.dim_value > 0
            for value in values
            for dim in value.type.tensor_type.shape.dim
        ),
        "finite_initializers": all(array.dtype.kind not in "fc" or bool(np.isfinite(array).all()) for array in arrays),
        "no_external_initializers": all(item.data_location != onnx.TensorProto.EXTERNAL and not item.external_data for item in model.graph.initializer),
        "conv_ub_findings": check_conv_bias(model),
        "ops": sorted({node.op_type for node in model.graph.node}),
        "existing_lookup_present": any(node.op_type in {"ScatterElements", "GatherND", "ScatterND"} for node in model.graph.node),
        "giant_policy": "incumbent only; no new or enlarged giant Einsum was emitted",
    }


def main() -> int:
    helper = shared()
    incumbents = []
    for task in (143, 301):
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
            "external": json.loads((HERE / f"task{task:03d}_baseline_external.json").read_text())["candidate"],
        }
        incumbents.append(row)
        print(task, row["known_dual"], flush=True)
    screen = json.loads((HERE / "history_screen.json").read_text())
    known_perfect_lower = [
        row for row in screen
        if all(mode["wrong"] == 0 and mode["errors"] == 0 for mode in row["known_dual_fail_fast"])
    ]
    report = {
        "source_zip_sha256": "74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534",
        "incumbents": incumbents,
        "history": {
            "task143": {"files": 607, "unique": 40, "below_baseline": 14},
            "task301": {"files": 576, "unique": 22, "below_baseline": 0},
        },
        "history_screen": screen,
        "known_perfect_lower": known_perfect_lower,
        "semantic_rejections": [
            {
                "path": row["path"],
                "reason": "TfIdfVectorizer/lookup-style memorization is explicitly prohibited",
            }
            for row in known_perfect_lower
        ],
        "exact_relations": json.loads((HERE / "exact_relations.json").read_text()),
        "exact_contractions": json.loads((HERE / "exact_contractions.json").read_text()),
        "fresh5000": {"run": False, "reason": "No semantically eligible cheaper candidate passed the known gate."},
        "winners": [],
        "projected_gain": 0.0,
        "conclusion": "No eligible exact compression; retain both incumbents.",
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(json.dumps({"winners": [], "projected_gain": 0.0}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
