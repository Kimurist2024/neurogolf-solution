#!/usr/bin/env python3
"""Exact no-bias feasibility census for task012's 8x8 depthwise Conv.

The scorer thresholds Conv output at raw > 0.  For each of the complete 196
generator states, this script asks whether a homogeneous linear classifier can
simultaneously implement the center and arm channels with one shared 8x8
kernel.  Infeasibility is accompanied by an exact-rational Farkas certificate.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / (
    "scripts/golf/root_task012_h8w8_policy90_272/candidates/"
    "task012_h8w8_policy90.onnx"
)
PARENT_SHA256 = "9aea31a6c01f7af21d893f6e5dde16dc947cdb17088686654f3f568845fbb947"
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
KERNEL = 8
PAD_TOP = PAD_LEFT = 3
EXPECTED_IO = (1, 10, 30, 30)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASK_DIR))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402

GEN = importlib.import_module("task_0962bcdd")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def role_channels(example: dict[str, Any]) -> tuple[int, int]:
    values = np.asarray(example["input"], dtype=np.int8)
    colors, counts = np.unique(values[values > 0], return_counts=True)
    center = int(colors[int(np.argmin(counts))])
    arm = int(colors[int(np.argmax(counts))])
    if sorted(counts.tolist()) != [2, 8]:
        raise RuntimeError(f"unexpected role counts: {counts.tolist()}")
    return center, arm


def encode(grid: list[list[int]], channel: int) -> np.ndarray:
    values = np.asarray(grid, dtype=np.int8)
    result = np.zeros((30, 30), dtype=np.uint8)
    result[: values.shape[0], : values.shape[1]] = values == channel
    return result


def patches(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, ((PAD_TOP, KERNEL - 1 - PAD_TOP), (PAD_LEFT, KERNEL - 1 - PAD_LEFT)))
    view = np.lib.stride_tricks.sliding_window_view(padded, (KERNEL, KERNEL))
    return np.ascontiguousarray(view.reshape(900, KERNEL * KERNEL), dtype=np.uint8)


def constraint_map(
    example: dict[str, Any], channels: tuple[int, ...]
) -> tuple[list[bytes], np.ndarray, np.ndarray]:
    seen: dict[bytes, int] = {}
    for channel in channels:
        inputs = patches(encode(example["input"], channel))
        outputs = encode(example["output"], channel).reshape(-1)
        for patch, label in zip(inputs, outputs, strict=True):
            key = np.packbits(patch, bitorder="big").tobytes()
            value = int(label)
            previous = seen.get(key)
            if previous is not None and previous != value:
                raise RuntimeError("one state contains an identical patch with contradictory labels")
            seen[key] = value
    keys = sorted(seen)
    unpacked = np.unpackbits(
        np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(len(keys), -1),
        axis=1,
        bitorder="big",
    )[:, : KERNEL * KERNEL].astype(np.int8)
    labels = np.asarray([seen[key] for key in keys], dtype=np.int8)
    return keys, unpacked, labels


def homogeneous_feasible(inputs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    # raw > 0: positive patches may be scaled to margin >=1; negatives allow raw==0.
    matrix = np.where(labels[:, None] > 0, -inputs, inputs).astype(np.float64)
    bounds = np.where(labels > 0, -1.0, 0.0)
    result = linprog(
        np.zeros(inputs.shape[1], dtype=np.float64),
        A_ub=matrix,
        b_ub=bounds,
        bounds=[(None, None)] * inputs.shape[1],
        method="highs",
    )
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "constraint_count": int(len(labels)),
        "positive_constraints": int(labels.sum()),
        "negative_constraints": int(len(labels) - labels.sum()),
    }


def farkas_certificate(
    keys: list[bytes], inputs: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Prove no w satisfies positives w*x>=1 and negatives w*x<=0."""
    signed = np.where(labels[:, None] > 0, -inputs, inputs).astype(np.int8)
    equalities = np.vstack(
        [signed.T.astype(np.float64), (labels > 0).astype(np.float64)]
    )
    target = np.concatenate([np.zeros(inputs.shape[1]), np.ones(1)])
    result = linprog(
        np.ones(len(labels), dtype=np.float64),
        A_eq=equalities,
        b_eq=target,
        bounds=[(0.0, None)] * len(labels),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"failed to obtain Farkas certificate: {result.message}")
    support = np.flatnonzero(result.x > 1e-9)
    coefficients = [
        Fraction(float(result.x[index])).limit_denominator(1_000_000)
        for index in support
    ]
    exact_zero = [
        sum(
            coefficient * int(signed[index, column])
            for index, coefficient in zip(support, coefficients, strict=True)
        )
        for column in range(inputs.shape[1])
    ]
    positive_mass = sum(
        coefficient
        for index, coefficient in zip(support, coefficients, strict=True)
        if labels[index] > 0
    )
    exact = bool(
        coefficients
        and all(coefficient >= 0 for coefficient in coefficients)
        and all(value == 0 for value in exact_zero)
        and positive_mass == 1
    )
    if not exact:
        raise RuntimeError("floating Farkas solution did not rationalize exactly")
    return {
        "exact_rational": True,
        "identity": (
            "nonnegative weighted sum of signed constraint rows is exactly zero; "
            "positive-row coefficient mass is one, hence y^T b=-1<0"
        ),
        "support_size": int(len(support)),
        "positive_coefficient_mass": str(positive_mass),
        "support": [
            {
                "constraint_index": int(index),
                "label": int(labels[index]),
                "patch_hex": keys[index].hex(),
                "coefficient": str(coefficient),
            }
            for index, coefficient in zip(support, coefficients, strict=True)
        ],
    }


def domain_states() -> list[tuple[tuple[int, int, int], dict[str, Any]]]:
    return [
        (
            (left, right, gravity),
            GEN.generate(colors=[1, 2], cols=[left, right], gravity=gravity),
        )
        for left in range(3, 10)
        for right in range(3, 10)
        for gravity in range(4)
    ]


def no_bias_probe(parent: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(parent)
    node = model.graph.node[0]
    if node.op_type != "Conv" or list(node.input) != ["input", "w", "b"]:
        raise RuntimeError("unexpected parent Conv")
    del node.input[2:]
    kept = [copy.deepcopy(item) for item in model.graph.initializer if item.name != "b"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.producer_name = "root_task012_h8w8_nobias_274_probe"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def profile(model: onnx.ModelProto) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="task012_nobias274_", dir="/tmp") as directory:
        path = Path(directory) / "probe.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def onehot(grid: list[list[int]]) -> np.ndarray:
    values = np.asarray(grid, dtype=np.int64)
    result = np.zeros(EXPECTED_IO, dtype=np.float32)
    rows, columns = np.indices(values.shape)
    result[0, values, rows, columns] = 1.0
    return result


def simple_drop_runtime(model: onnx.ModelProto, states: list[tuple[tuple[int, int, int], dict[str, Any]]]) -> dict[str, Any]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitizer rejected no-bias probe")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    right = errors = nonfinite = shape_mismatch = 0
    for _state, example in states:
        try:
            raw = session.run(["output"], {"input": onehot(example["input"])})[0]
        except Exception:
            errors += 1
            continue
        nonfinite += int(not np.isfinite(raw).all())
        shape_mismatch += int(tuple(raw.shape) != EXPECTED_IO)
        right += int(np.array_equal(raw > 0, onehot(example["output"]) > 0))
    return {
        "right": right,
        "total": len(states),
        "rate": right / len(states),
        "errors": errors,
        "nonfinite_cases": nonfinite,
        "output_shape_mismatches": shape_mismatch,
    }


def main() -> None:
    if sha256(PARENT) != PARENT_SHA256:
        raise RuntimeError("cost650 parent SHA mismatch")
    parent = onnx.load(PARENT)
    probe = no_bias_probe(parent)
    parent_profile = profile(parent)
    probe_profile = profile(probe)
    if parent_profile != {"memory": 0, "params": 650, "cost": 650}:
        raise RuntimeError(f"unexpected parent profile: {parent_profile}")
    if probe_profile != {"memory": 0, "params": 640, "cost": 640}:
        raise RuntimeError(f"unexpected no-bias profile: {probe_profile}")

    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item)) for item in probe.graph.initializer
    }
    weights = arrays["w"]
    nonzero_equal = all(weights[index].tobytes() == weights[1].tobytes() for index in range(2, 10))
    if not nonzero_equal:
        raise RuntimeError("parent nonzero kernels are not byte-identical")

    states = domain_states()
    rows = []
    for state, example in states:
        center, arm = role_channels(example)
        background_keys, background_x, background_y = constraint_map(example, (0,))
        foreground_keys, foreground_x, foreground_y = constraint_map(example, (center, arm))
        background = homogeneous_feasible(background_x, background_y)
        foreground = homogeneous_feasible(foreground_x, foreground_y)
        certificate = None
        if not foreground["success"]:
            certificate = farkas_certificate(foreground_keys, foreground_x, foreground_y)
        rows.append(
            {
                "state": list(state),
                "center_channel": center,
                "arm_channel": arm,
                "background": background,
                "foreground_shared_kernel": foreground,
                "case_individually_feasible": bool(background["success"] and foreground["success"]),
                "foreground_farkas_certificate": certificate,
                "background_constraint_digest": sha256_bytes(
                    b"".join(background_keys) + background_y.tobytes()
                ),
                "foreground_constraint_digest": sha256_bytes(
                    b"".join(foreground_keys) + foreground_y.tobytes()
                ),
            }
        )

    background_feasible = sum(row["background"]["success"] for row in rows)
    foreground_feasible = sum(row["foreground_shared_kernel"]["success"] for row in rows)
    individual_case_upper = sum(row["case_individually_feasible"] for row in rows)
    policy90_required = math.ceil(0.90 * len(rows))
    exact_certificates = sum(
        bool(row["foreground_farkas_certificate"])
        and row["foreground_farkas_certificate"]["exact_rational"]
        for row in rows
    )
    if individual_case_upper != 0 or exact_certificates != len(rows):
        raise RuntimeError("unexpected no-bias feasibility result")

    result = {
        "task": 12,
        "lane": "root_task012_h8w8_nobias_274",
        "decision": "NO_POLICY90_NOBIAS_CANDIDATE",
        "parent": {
            "path": str(PARENT.relative_to(ROOT)),
            "sha256": sha256(PARENT),
            "profile": parent_profile,
        },
        "hypothetical_no_bias_family": {
            "graph": "one output-only group10 Conv; weights [10,1,8,8]; no bias input/initializer",
            "profile": probe_profile,
            "padding": [3, 3, 4, 4],
            "nonzero_channels_1_to_9_share_one_kernel": True,
            "background_kernel_independent": True,
            "probe_serialized_sha256": sha256_bytes(probe.SerializeToString()),
            "probe_conv_bias_findings": check_conv_bias(probe),
            "simple_parent_bias_drop_runtime_domain196": simple_drop_runtime(probe, states),
        },
        "complete_domain": {
            "derivation": "7 col0 values * 7 col1 values * 4 gravity values",
            "colors_representative": [1, 2],
            "states": len(rows),
            "state_tuple_sha256": sha256_bytes(
                np.asarray([state for state, _example in states], dtype=np.int16).tobytes()
            ),
            "background_individually_feasible_states": background_feasible,
            "foreground_individually_feasible_states": foreground_feasible,
            "case_individually_feasible_states": individual_case_upper,
            "case_level_optimal_upper_bound": individual_case_upper,
            "case_level_optimal_upper_rate": individual_case_upper / len(rows),
            "policy90_required_cases": policy90_required,
            "all_foreground_infeasibilities_have_exact_rational_farkas_certificates": (
                exact_certificates == len(rows)
            ),
            "proof": (
                "Every one of the 196 states is individually infeasible for the shared center/arm "
                "homogeneous classifier. Therefore no common foreground kernel, regardless of "
                "the independent background kernel, can make even one complete case exact."
            ),
        },
        "state_rows": rows,
        "candidate": None,
        "winner": None,
        "candidate_runtime_gate": {
            "run": False,
            "skip_reason": "Exact case-level upper bound 0/196 is below POLICY90's 177/196 requirement.",
            "known265": False,
            "domain196": False,
            "fresh_two_by_10000_four_configs": False,
            "full_strict_truthful_static_standard_finite_ub0_error0_shape0": False,
        },
        "policy": {
            "lookup": False,
            "fixture_correction": False,
            "shape_cloak": False,
            "private_zero_candidate": False,
            "root_or_71407_modified": False,
        },
    }
    (HERE / "search.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "candidates.json").write_text(
        json.dumps(
            {
                "task": 12,
                "parent_sha256": PARENT_SHA256,
                "hypothetical_cost": 640,
                "candidates": [],
                "winner": None,
                "skip_reason": result["candidate_runtime_gate"]["skip_reason"],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "parent_profile": parent_profile,
                "hypothetical_profile": probe_profile,
                "background_individually_feasible": background_feasible,
                "foreground_individually_feasible": foreground_feasible,
                "case_level_optimal_upper_bound": individual_case_upper,
                "policy90_required": policy90_required,
                "exact_farkas_certificates": exact_certificates,
                "output": str((HERE / "search.json").relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
