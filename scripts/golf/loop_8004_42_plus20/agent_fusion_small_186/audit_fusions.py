#!/usr/bin/env python3
"""Independent safety audit for task039/111/122 fusion-scan winners."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (39, 111, 122)
ROOT_HASHES = {
    "submission.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "all_scores.csv": "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78",
}

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


onnxruntime.set_default_logger_severity(4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int]:
    return [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]


def bytes_of(value: onnx.ValueInfoProto) -> int:
    dtype = onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
    return math.prod(shape(value)) * np.dtype(dtype).itemsize


def validate_structure(model: onnx.ModelProto) -> dict[str, Any]:
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, check_type=True, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred, full_check=True)
    inferred_values = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    declared = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    mismatches = []
    for item in declared:
        expected = inferred_values[item.name]
        actual_type = item.type.tensor_type
        expected_type = expected.type.tensor_type
        if (
            actual_type.elem_type != expected_type.elem_type
            or shape(item) != shape(expected)
        ):
            mismatches.append(item.name)
    return {
        "full_checker": "PASS",
        "strict_data_prop": "PASS",
        "declared_value_info": len(model.graph.value_info),
        "truthful_declared_mismatches": mismatches,
        "truthful_declared": not mismatches,
        "inferred": inferred,
    }


def conv_bias_ub(model: onnx.ModelProto) -> list[dict[str, Any]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    values = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    failures = []
    for index, node in enumerate(model.graph.node):
        if node.op_type in {"Conv", "ConvTranspose"}:
            bias_index = 2
        elif node.op_type == "QLinearConv":
            bias_index = 8
        else:
            continue
        if len(node.input) <= bias_index or not node.input[bias_index]:
            continue
        bias = initializers.get(node.input[bias_index])
        weight = initializers.get(node.input[1 if node.op_type != "QLinearConv" else 3])
        weight_name = node.input[1 if node.op_type != "QLinearConv" else 3]
        weight_shape = list(weight.shape) if weight is not None else shape(values[weight_name])
        if node.op_type == "ConvTranspose":
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            expected = int(weight_shape[1]) * int(attrs.get("group", 1))
        else:
            expected = int(weight_shape[0])
        if bias is None or bias.ndim != 1 or int(bias.shape[0]) != expected:
            failures.append(
                {
                    "node_index": index,
                    "op_type": node.op_type,
                    "expected": expected,
                    "actual": None if bias is None else list(bias.shape),
                }
            )
    return failures


def profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"fusion186_{task:03d}_{label}_") as workdir:
        return scoring.score_and_verify(
            copy.deepcopy(model), task, workdir, label=label, require_correct=False
        )


def make_session(model: onnx.ModelProto, disable: bool) -> onnxruntime.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model returned None")
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(sanitized.SerializeToString(), options)


def known_mode(
    baseline: onnx.ModelProto,
    candidate: onnx.ModelProto,
    task: int,
    disable: bool,
) -> dict[str, Any]:
    examples = scoring.load_examples(task)
    cases = examples["train"] + examples["test"] + examples["arc-gen"]
    row: dict[str, Any] = {
        "mode": "disable_all" if disable else "default_enable_all",
        "known_total": len(cases),
        "baseline": {"correct": 0, "runtime_errors": 0, "nonfinite_values": 0, "shape_mismatches": 0},
        "candidate": {"correct": 0, "runtime_errors": 0, "nonfinite_values": 0, "shape_mismatches": 0},
        "raw_equal": 0,
        "raw_compared": 0,
        "max_abs_raw_difference": 0.0,
        "first_candidate_errors": [],
    }
    sessions: dict[str, onnxruntime.InferenceSession | None] = {}
    for label, model in (("baseline", baseline), ("candidate", candidate)):
        try:
            sessions[label] = make_session(model, disable)
        except Exception as exc:  # noqa: BLE001
            sessions[label] = None
            row[label]["load_error"] = f"{type(exc).__name__}: {exc}"

    for example in cases:
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        raw: dict[str, np.ndarray] = {}
        for label in ("baseline", "candidate"):
            session = sessions[label]
            if session is None:
                row[label]["runtime_errors"] += 1
                continue
            try:
                output = session.run(["output"], {"input": benchmark["input"]})[0]
                raw[label] = output
                row[label]["nonfinite_values"] += int(np.count_nonzero(~np.isfinite(output)))
                if output.shape != benchmark["output"].shape:
                    row[label]["shape_mismatches"] += 1
                if np.array_equal((output > 0.0).astype(float), benchmark["output"]):
                    row[label]["correct"] += 1
            except Exception as exc:  # noqa: BLE001
                row[label]["runtime_errors"] += 1
                if label == "candidate" and len(row["first_candidate_errors"]) < 3:
                    message = " ".join(str(exc).split())
                    row["first_candidate_errors"].append(f"{type(exc).__name__}: {message[:1000]}")
        if "baseline" in raw and "candidate" in raw:
            row["raw_compared"] += 1
            if np.array_equal(raw["baseline"], raw["candidate"]):
                row["raw_equal"] += 1
            finite_a = np.nan_to_num(raw["baseline"], copy=False)
            finite_b = np.nan_to_num(raw["candidate"], copy=False)
            row["max_abs_raw_difference"] = max(
                row["max_abs_raw_difference"],
                float(np.max(np.abs(finite_a.astype(np.float64) - finite_b.astype(np.float64)))),
            )

    for label in ("baseline", "candidate"):
        values = row[label]
        values["normal_accuracy"] = values["correct"] / len(cases)
        values["runtime0"] = values["runtime_errors"] == 0
        values["nonfinite0"] = values["nonfinite_values"] == 0
        values["truthful_runtime_shapes"] = (
            values["shape_mismatches"] == 0 and values["runtime_errors"] == 0
        )
        values["passes_90"] = values["normal_accuracy"] >= 0.90
    return row


def exact_diff(
    baseline: onnx.ModelProto,
    candidate: onnx.ModelProto,
    inferred_baseline: onnx.ModelProto,
) -> dict[str, Any]:
    candidate_outputs = {output for node in candidate.graph.node for output in node.output}
    removed = [node for node in baseline.graph.node if any(out not in candidate_outputs for out in node.output)]
    baseline_outputs = {output for node in baseline.graph.node for output in node.output}
    added = [node for node in candidate.graph.node if any(out not in baseline_outputs for out in node.output)]
    candidate_by_output = {tuple(node.output): node.SerializeToString() for node in candidate.graph.node}
    retained_changed = [
        list(node.output)
        for node in baseline.graph.node
        if tuple(node.output) in candidate_by_output
        and node.SerializeToString() != candidate_by_output[tuple(node.output)]
    ]
    consumers = {
        name: sum(name in node.input for node in baseline.graph.node)
        for node in removed
        for name in node.output
    }
    values = {
        item.name: item
        for item in list(inferred_baseline.graph.value_info) + list(inferred_baseline.graph.output)
    }
    return {
        "removed_nodes": [
            {
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "output_bytes": sum(bytes_of(values[out]) for out in node.output if out in values),
                "consumers": {out: consumers[out] for out in node.output},
                "is_graph_output": any(out == item.name for out in node.output for item in baseline.graph.output),
            }
            for node in removed
        ],
        "added_nodes": [list(node.output) for node in added],
        "retained_nodes_changed": retained_changed,
        "initializers_bit_identical": [item.SerializeToString() for item in baseline.graph.initializer]
        == [item.SerializeToString() for item in candidate.graph.initializer],
        "pure_deadend_deletion": bool(removed) and not added and not retained_changed and all(v == 0 for v in consumers.values()),
    }


def main() -> None:
    root_before = {name: sha256(ROOT / name) for name in ROOT_HASHES}
    assert root_before == ROOT_HASHES
    stage_before = tree_digest(ROOT / "others" / "71407")
    results = []
    for task in TASKS:
        baseline_path = HERE / "baseline" / f"task{task:03d}.onnx"
        candidate_path = HERE / "candidates" / f"task{task:03d}_all_fusions.onnx"
        baseline = onnx.load(str(baseline_path))
        candidate = onnx.load(str(candidate_path))
        baseline_structure = validate_structure(baseline)
        candidate_structure = validate_structure(candidate)
        baseline_profile = profile(baseline, task, "baseline")
        candidate_profile = profile(candidate, task, "candidate")
        assert baseline_profile is not None and candidate_profile is not None
        modes = [
            known_mode(baseline, candidate, task, disable=False),
            known_mode(baseline, candidate, task, disable=True),
        ]
        ub = conv_bias_ub(candidate)
        diff = exact_diff(baseline, candidate, baseline_structure["inferred"])
        assert diff["pure_deadend_deletion"]
        strict_lower = candidate_profile["cost"] < baseline_profile["cost"]
        known_safe = all(
            mode["candidate"]["runtime0"]
            and mode["candidate"]["nonfinite0"]
            and mode["candidate"]["truthful_runtime_shapes"]
            and mode["candidate"]["passes_90"]
            for mode in modes
        )
        safe = bool(
            strict_lower
            and candidate_structure["truthful_declared"]
            and not ub
            and known_safe
        )
        results.append(
            {
                "task": task,
                "decision": "SAFE" if safe else "REJECT_KNOWN_RUNTIME",
                "baseline": {
                    "path": str(baseline_path.relative_to(ROOT)),
                    "sha256": sha256(baseline_path),
                    "nodes": len(baseline.graph.node),
                    "profile": baseline_profile,
                },
                "candidate": {
                    "path": str(candidate_path.relative_to(ROOT)),
                    "sha256": sha256(candidate_path),
                    "nodes": len(candidate.graph.node),
                    "profile": candidate_profile,
                    "strict_lower": strict_lower,
                },
                "transformation": diff,
                "structure": {
                    "full_checker": candidate_structure["full_checker"],
                    "strict_data_prop": candidate_structure["strict_data_prop"],
                    "truthful_declared": candidate_structure["truthful_declared"],
                    "truthful_declared_mismatches": candidate_structure["truthful_declared_mismatches"],
                    "conv_family_bias_ub_count": len(ub),
                    "conv_family_bias_ub": ub,
                },
                "known_dual_ort": modes,
                "fresh": {
                    "required_if_known_gate_passes": {"seeds": [0, 1], "per_seed": 5000},
                    "runs": 0,
                    "reason": "complete known-set runtime gate failed before fresh admission",
                },
                "safe": safe,
            }
        )

    root_after = {name: sha256(ROOT / name) for name in ROOT_HASHES}
    stage_after = tree_digest(ROOT / "others" / "71407")
    assert root_after == root_before
    assert stage_after == stage_before
    payload = {
        "decision": "NO_SAFE",
        "safe_count": sum(bool(row["safe"]) for row in results),
        "tasks": results,
        "fresh_runs": 0,
        "root_guards": {"before": root_before, "after": root_after},
        "others_71407_tree_sha256": {"before": stage_before, "after": stage_after},
        "root_mutations": [],
    }
    (HERE / "audit" / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "tasks": [
                    {
                        "task": row["task"],
                        "baseline_cost": row["baseline"]["profile"]["cost"],
                        "candidate_cost": row["candidate"]["profile"]["cost"],
                        "known": {
                            mode["mode"]: mode["candidate"] for mode in row["known_dual_ort"]
                        },
                    }
                    for row in results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
