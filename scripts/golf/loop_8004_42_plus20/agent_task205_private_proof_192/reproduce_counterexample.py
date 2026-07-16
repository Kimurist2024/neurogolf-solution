#!/usr/bin/env python3
"""Reproduce a generator-reachable task205 counterexample for the cost-937 net."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "inputs/arc-gen-repo/tasks")]
from lib import scoring  # noqa: E402


SEED = 93_023_205
CASE = 11
EXPECTED = {
    "lead937": "bbfa8f5b79d2e8345a39a41f327ac1c2c851f3c7f388dd595c72ef951e1b3050",
    "authority1042": "8a6acdc20a366ccbd32cf761285cbb2f1cbcf7d3d2ef8ea71d0fb5a3ed6f1468",
    "staged_exact1041": "509c1947929ab888cff4443ac5b6d808b213fa5057e1c03a2758c1717b3f9eed",
    "authority_rewrite1038": "43c963c46bda5b444fb830b5495b4d71fb9dcf958e108954cdb9ef1064d9f9a8",
}
MODELS = {
    "lead937": ROOT / "scripts/golf/loop_7999_13/lane_a23/candidates/task205_r02.onnx",
    "authority1042": ROOT / "scripts/golf/loop_7999_13/lane_a23/baseline/task205.onnx",
    "staged_exact1041": ROOT / "scripts/golf/loop_8004_42_plus20/agent_high205_338_123/candidates/task205_rowpow_selu.onnx",
    "authority_rewrite1038": ROOT / "scripts/golf/loop_7999_13/lane_a23/candidates/task205_c3_cost1038.onnx",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_session(model: onnx.ModelProto, disable: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def run(model: onnx.ModelProto, feed: np.ndarray, disable: bool) -> np.ndarray:
    session = make_session(model, disable)
    return np.asarray(
        session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: feed},
        )[0]
    )


def trace_all(model: onnx.ModelProto, feed: np.ndarray, disable: bool) -> dict[str, Any]:
    """Expose every graph value and check runtime shape/finite-value truthfulness."""
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model failed")
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(sanitized), strict_mode=True, data_prop=True
    )
    infos = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    expected_shapes = {
        name: [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]
        for name, value in infos.items()
    }
    existing = {value.name for value in sanitized.graph.output}
    for node in sanitized.graph.node:
        for name in node.output:
            if name and name not in existing:
                sanitized.graph.output.append(copy.deepcopy(infos[name]))
                existing.add(name)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    values = session.run(None, {session.get_inputs()[0].name: feed})
    names = [value.name for value in session.get_outputs()]
    mismatches = [
        {"name": name, "declared": expected_shapes[name], "actual": list(value.shape)}
        for name, value in zip(names, values)
        if expected_shapes[name] != list(value.shape)
    ]
    nonfinite = sum(
        int(value.size - np.count_nonzero(np.isfinite(value)))
        for value in values
        if np.issubdtype(value.dtype, np.number)
    )
    return {
        "runtime_tensor_count": len(values),
        "declared_actual_shape_mismatches": mismatches,
        "nonfinite_values": nonfinite,
        "runtime_bytes": sum(int(value.nbytes) for value in values),
    }


def reconstruct_latent(example: dict[str, Any]) -> dict[str, Any]:
    """Parse this fixed witness's planted rectangle and provide explicit args.

    The values are also independently checked against the random-call transcript
    below.  Supplying the flattened final input as ``colors`` reproduces the same
    example and is itself a positive-probability color draw before overwriting.
    """
    latent = {
        "width": 15,
        "height": 30,
        "wide": 9,
        "tall": 6,
        "rowoffset": 4,
        "coloffset": 2,
        "rows": [4, 3, 1],
        "cols": [5, 6, 2],
        "marker_color": 3,
        "box_color": 8,
        "colors": [value for row in example["input"] for value in row],
    }
    return latent


def support_checks(latent: dict[str, Any]) -> dict[str, bool]:
    return {
        "grid_dimensions": 15 <= latent["width"] <= 30 and 15 <= latent["height"] <= 30,
        "box_dimensions": 6 <= latent["wide"] <= 10 and 6 <= latent["tall"] <= 10,
        "rowoffset": 1 <= latent["rowoffset"] <= latent["height"] - latent["tall"] - 1,
        "coloffset": 1 <= latent["coloffset"] <= latent["width"] - latent["wide"] - 1,
        "marker_count": 1 <= len(latent["rows"]) == len(latent["cols"]) <= 3,
        "distinct_interior_rows": len(set(latent["rows"])) == len(latent["rows"])
        and all(1 <= value < latent["tall"] - 1 for value in latent["rows"]),
        "distinct_interior_cols": len(set(latent["cols"])) == len(latent["cols"])
        and all(1 <= value < latent["wide"] - 1 for value in latent["cols"]),
        "distinct_colors": latent["marker_color"] != latent["box_color"],
        "colors_in_domain": all(0 <= value <= 9 for value in latent["colors"]),
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    module = importlib.import_module("task_8731374e")
    random.seed(SEED)
    np.random.seed(SEED & 0xFFFFFFFF)
    example = None
    for _ in range(CASE):
        example = module.generate()
    assert example is not None
    latent = reconstruct_latent(example)
    checks = support_checks(latent)
    explicit = module.generate(
        width=latent["width"],
        height=latent["height"],
        wide=latent["wide"],
        tall=latent["tall"],
        rowoffset=latent["rowoffset"],
        coloffset=latent["coloffset"],
        rows=latent["rows"],
        cols=latent["cols"],
        colors=latent["colors"],
    )
    checks["explicit_generate_reproduces_seed_case"] = explicit == example
    if not all(checks.values()):
        raise AssertionError(checks)

    benchmark = scoring.convert_to_numpy(example)
    if benchmark is None:
        raise RuntimeError("witness conversion failed")
    expected = benchmark["output"] > 0
    model_results: dict[str, Any] = {}
    for label, path in MODELS.items():
        actual_sha = digest(path)
        if actual_sha != EXPECTED[label]:
            raise RuntimeError(f"{label} SHA drift: {actual_sha}")
        model = onnx.load(path)
        mode_results: dict[str, Any] = {}
        for disable, mode in ((True, "disable_all"), (False, "default")):
            raw = run(model, benchmark["input"], disable)
            decoded = raw > 0
            indices = np.argwhere(decoded != expected)
            mode_results[mode] = {
                "matches_generator_gold": bool(np.array_equal(decoded, expected)),
                "different_onehot_cells": int(indices.shape[0]),
                "different_cells": [
                    {
                        "index": [int(value) for value in index],
                        "expected": bool(expected[tuple(index)]),
                        "predicted": bool(decoded[tuple(index)]),
                        "raw": float(raw[tuple(index)]),
                    }
                    for index in indices
                ],
                "output_shape": list(raw.shape),
                "nonfinite_output_values": int(raw.size - np.count_nonzero(np.isfinite(raw))),
                "all_intermediate_trace": trace_all(model, benchmark["input"], disable),
            }
        model_results[label] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": actual_sha,
            "modes": mode_results,
        }

    h, w = latent["tall"], latent["wide"]
    lead_raw = run(onnx.load(MODELS["lead937"]), benchmark["input"], True)
    lead_grid = np.argmax(lead_raw[0, :, :h, :w] > 0, axis=0)
    result = {
        "task": 205,
        "generator": "inputs/arc-gen-repo/tasks/task_8731374e.py",
        "generator_sha256": digest(ROOT / "inputs/arc-gen-repo/tasks/task_8731374e.py"),
        "seed": SEED,
        "valid_case_one_based": CASE,
        "latent": latent,
        "support_checks": checks,
        "input_grid": example["input"],
        "gold_output_grid": example["output"],
        "lead937_decoded_output_grid": lead_grid.tolist(),
        "lead_failure_summary": {
            "correct_box_location": True,
            "correct_box_color": 8,
            "correct_marker_color": 3,
            "missed_relative_marker_row": 1,
            "wrong_decoded_pixels": 6,
            "different_onehot_cells": 12,
        },
        "models": model_results,
        "disposition": "HARD_REJECT_EXPLICIT_GENERATOR_REACHABLE_COUNTEREXAMPLE",
    }
    (HERE / "counterexample.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "lead_sha256": EXPECTED["lead937"],
        "seed": SEED,
        "case": CASE,
        "support": all(checks.values()),
        "lead_disable_all_diff": model_results["lead937"]["modes"]["disable_all"]["different_onehot_cells"],
        "lead_default_diff": model_results["lead937"]["modes"]["default"]["different_onehot_cells"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
