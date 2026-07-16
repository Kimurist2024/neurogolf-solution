#!/usr/bin/env python3
"""Assemble structural, dual-runtime, history, and external C32 evidence."""

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
    spec = importlib.util.spec_from_file_location("c32_shared", path)
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
        "giant_policy": "incumbent grandfathered; no candidate enlarged an Einsum",
    }


def main() -> int:
    helper = shared()
    incumbents = []
    for task in (224, 240):
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        model = onnx.load(path)
        memory, params, total = cost_of(path)
        external_path = (
            HERE / "task224_baseline_external.json"
            if task == 224
            else HERE / "task240_candidate_external.json"
        )
        external = json.loads(external_path.read_text())
        external_row = external["candidate"] if task == 224 else external["baseline"]
        row = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path),
            "cost": {"memory": memory, "params": params, "total": total},
            "structure": structure(model),
            "runtime_shapes": helper.trace_runtime_shapes(copy.deepcopy(model), task),
            "known_dual": helper.known_dual(copy.deepcopy(model), task),
            "external": external_row,
        }
        incumbents.append(row)
        print(task, row["known_dual"], flush=True)

    relation = json.loads((HERE / "exact_relations.json").read_text())
    screen = json.loads((HERE / "candidate_screen.json").read_text())
    task240_attempt = json.loads((HERE / "task240_candidate_external.json").read_text())
    report = {
        "source_zip": "scripts/golf/loop_7999_13/submission_7999.13_wave17_candidate_meta.zip",
        "source_zip_sha256": "74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534",
        "incumbents": incumbents,
        "history": {
            "task224": {"files": 578, "unique": 18, "below_baseline": 4},
            "task240": {"files": 591, "unique": 26, "below_baseline": 5},
        },
        "historical_cheaper_screen": screen,
        "exact_relation_scan": relation,
        "attempts": {
            "task224_csum_cdiag_tie": {
                "candidate_emitted": False,
                "reason": (
                    "Csum and Cdiag are diagonally equivalent, but the global magnitude gauge "
                    "system is inconsistent even when row_codes/col_codes diagonal output gauges are allowed."
                ),
            },
            "task240_absorb_a3": {
                "path": "scripts/golf/loop_7999_13/lane_c32/task240_absorb_a3.onnx",
                "sha256": sha(HERE / "task240_absorb_a3.onnx"),
                "cost": 170,
                "external_known": task240_attempt["candidate"]["known"],
                "reason": (
                    "Rejected: two B occurrences have no paired 2-vector factor, so the proposed "
                    "global row gauge changes their row sums; known 0/266, errors 0."
                ),
            },
        },
        "fresh5000": {
            "run": False,
            "reason": "No cheaper candidate passed the complete known gate in both ORT modes.",
        },
        "winners": [],
        "projected_gain": 0.0,
        "conclusion": "No eligible strict reduction; retain task224 and task240 incumbents.",
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(
        json.dumps({"winners": [], "projected_gain": 0.0}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
