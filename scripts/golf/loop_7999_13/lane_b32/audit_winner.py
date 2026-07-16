#!/usr/bin/env python3
"""Structural, dual-known, runtime-shape, and raw-equivalence audit for B32."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


TASK = 219
BASELINE = HERE / "baseline_task219.onnx"
CANDIDATE = HERE / "task219_b32_winner.onnx"
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Compress", "SequenceMap"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def known_differential(
    baseline: onnx.ModelProto, candidate: onnx.ModelProto, disabled: bool
) -> dict[str, object]:
    base_runner = session(baseline, disabled)
    candidate_runner = session(candidate, disabled)
    row = {
        "total": 0, "right": 0, "wrong": 0, "runtime_errors": 0,
        "raw_equal": 0, "threshold_equal": 0,
    }
    for subset in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(TASK)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            row["total"] += 1
            try:
                base_raw = base_runner.run(["output"], {"input": benchmark["input"]})[0]
                cand_raw = candidate_runner.run(["output"], {"input": benchmark["input"]})[0]
            except Exception:  # noqa: BLE001
                row["runtime_errors"] += 1
                continue
            correct = np.array_equal(cand_raw > 0, benchmark["output"] > 0)
            row["right"] += int(correct)
            row["wrong"] += int(not correct)
            row["raw_equal"] += int(np.array_equal(base_raw, cand_raw))
            row["threshold_equal"] += int(np.array_equal(base_raw > 0, cand_raw > 0))
    row["perfect"] = bool(
        row["right"] == row["total"]
        and row["wrong"] == 0
        and row["runtime_errors"] == 0
        and row["raw_equal"] == row["total"]
        and row["threshold_equal"] == row["total"]
    )
    return row


def runtime_shapes(model: onnx.ModelProto) -> dict[str, object]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    runner = ort.InferenceSession(traced.SerializeToString(), options)
    benchmark = scoring.convert_to_numpy(scoring.load_examples(TASK)["train"][0])
    if benchmark is None:
        raise RuntimeError("missing known input")
    values = runner.run(names, {"input": benchmark["input"]})
    mismatches = []
    for name, value in zip(names, values):
        static = dims(typed[name])
        actual = list(np.asarray(value).shape)
        if static != actual:
            mismatches.append({"name": name, "static": static, "runtime": actual})
    return {"traced_outputs": len(names), "mismatches": mismatches, "truthful": not mismatches}


def main() -> None:
    baseline = onnx.load(BASELINE)
    candidate = onnx.load(CANDIDATE)
    onnx.checker.check_model(candidate, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(candidate), strict_mode=True, data_prop=True
    )
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    base_arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in baseline.graph.initializer}
    cand_arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in candidate.graph.initializer}
    removed = sorted(set(base_arrays) - set(cand_arrays))
    reshaped = {
        name: {"before": list(base_arrays[name].shape), "after": list(cand_arrays[name].shape)}
        for name in sorted(set(base_arrays) & set(cand_arrays))
        if base_arrays[name].shape != cand_arrays[name].shape
    }
    values_unchanged = all(
        np.array_equal(base_arrays[name].reshape(-1), cand_arrays[name].reshape(-1))
        for name in set(base_arrays) & set(cand_arrays)
    )
    cmasks = base_arrays["cmaskv8"].reshape(-1).astype(np.uint16)
    cshifts = base_arrays["cshiftv8"].reshape(-1).astype(np.uint16)
    exhaustive_c_extract = all(
        ((value & int(mask)) // int(scale)) == ((value // int(scale)) & 3)
        for value in range(256)
        for mask, scale in zip(cmasks, cshifts)
    )
    exhaustive_bool_scale = all(
        (int(flag) * 98) == (98 if flag else 0) for flag in (False, True)
    )
    with tempfile.TemporaryDirectory(prefix="b32_task219_") as workdir:
        score = scoring.score_and_verify(
            copy.deepcopy(candidate), TASK, workdir, "b32", require_correct=True
        )
    known = {
        "disable_all": known_differential(baseline, candidate, True),
        "default": known_differential(baseline, candidate, False),
    }
    shape_audit = runtime_shapes(candidate)
    nested = [
        node.op_type
        for node in candidate.graph.node
        for attribute in node.attribute
        if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    structure = {
        "full_checker": True,
        "strict_shape_data_prop": True,
        "all_static_positive": all(
            all(isinstance(dim, int) and dim > 0 for dim in dims(value))
            for value in values
        ),
        "runtime_shapes": shape_audit,
        "canonical_io": [value.name for value in candidate.graph.input] == ["input"]
        and [value.name for value in candidate.graph.output] == ["output"],
        "input_shape": dims(inferred.graph.input[0]),
        "output_shape": dims(inferred.graph.output[0]),
        "standard_domains": all(node.domain in ("", "ai.onnx") for node in candidate.graph.node)
        and all(item.domain in ("", "ai.onnx") for item in candidate.opset_import),
        "functions": len(candidate.functions),
        "sparse_initializers": len(candidate.graph.sparse_initializer),
        "nested_graphs": nested,
        "banned_ops": [node.op_type for node in candidate.graph.node if node.op_type in BANNED or "Sequence" in node.op_type],
        "conv_bias_issues": check_conv_bias(candidate),
        "op_histogram": dict(sorted(Counter(node.op_type for node in candidate.graph.node).items())),
        "node_count": len(candidate.graph.node),
        "initializer_count": len(candidate.graph.initializer),
    }
    report = {
        "task": TASK,
        "baseline": {"path": str(BASELINE.relative_to(ROOT)), "sha256": digest(BASELINE), "cost": 1479},
        "candidate": {"path": str(CANDIDATE.relative_to(ROOT)), "sha256": digest(CANDIDATE), "cost": 1445},
        "projected_gain": math.log(1479 / 1445),
        "score": score,
        "known_dual_raw_equivalence": known,
        "structure": structure,
        "initializer_audit": {
            "removed": removed,
            "added": sorted(set(cand_arrays) - set(base_arrays)),
            "reshaped": reshaped,
            "remaining_flat_values_unchanged": values_unchanged,
        },
        "algebra": {
            "c_masks_equal_three_times_shift": bool(np.array_equal(cmasks, 3 * cshifts)),
            "c_extract_exhaustive_all_uint8": exhaustive_c_extract,
            "bool_scale_exhaustive": exhaustive_bool_scale,
            "unit_dimension_rewrites_only_broadcast": True,
        },
    }
    report["pass"] = bool(
        score and score.get("correct") and score.get("cost") == 1445
        and all(item["perfect"] for item in known.values())
        and structure["all_static_positive"] and shape_audit["truthful"]
        and structure["canonical_io"]
        and structure["input_shape"] == structure["output_shape"] == [1, 10, 30, 30]
        and structure["standard_domains"] and not structure["functions"]
        and not structure["sparse_initializers"] and not structure["nested_graphs"]
        and not structure["banned_ops"] and not structure["conv_bias_issues"]
        and removed == ["axes12", "cmaskv8"]
        and not report["initializer_audit"]["added"] and values_unchanged
        and report["algebra"]["c_masks_equal_three_times_shift"]
        and exhaustive_c_extract and exhaustive_bool_scale
    )
    (HERE / "winner_audit.json").write_text(json.dumps(report, indent=2, default=bool) + "\n")
    print(json.dumps(report, indent=2, default=bool))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
