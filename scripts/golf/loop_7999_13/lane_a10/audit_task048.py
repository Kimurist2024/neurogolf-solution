#!/usr/bin/env python3
"""Strict dual-ORT and independent fresh-5000 audit for the A10 task048 candidate."""

from __future__ import annotations

import collections
import copy
import hashlib
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402
from verify_fix import official_gold  # noqa: E402


TASK = 48
TASK_HASH = "239be575"
COUNT = 5000
BASELINE = HERE / "baseline" / "task048.onnx"
CANDIDATE = HERE / "task048_fold_singleton_axes.onnx"
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_session(path: Path, disable_all: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known(session: ort.InferenceSession) -> dict[str, object]:
    examples = scoring.load_examples(TASK)
    subsets: dict[str, dict[str, int]] = {}
    for name in ("train", "test", "arc-gen"):
        right = wrong = errors = 0
        for example in examples[name]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                actual = session.run(["output"], {"input": benchmark["input"]})[0] > 0.0
                if np.array_equal(actual, benchmark["output"] > 0.0):
                    right += 1
                else:
                    wrong += 1
            except Exception:  # noqa: BLE001 - errors are audit data
                errors += 1
        subsets[name] = {"right": right, "wrong": wrong, "errors": errors}
    subsets["total"] = {
        key: sum(int(subsets[name][key]) for name in ("train", "test", "arc-gen"))
        for key in ("right", "wrong", "errors")
    }
    return subsets


def fresh(disable_all: bool, seed: int) -> dict[str, object]:
    baseline = make_session(BASELINE, disable_all)
    candidate = make_session(CANDIDATE, disable_all)
    generator = importlib.import_module(f"task_{TASK_HASH}")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    start = time.monotonic()
    attempts = generation_errors = oversize = valid = 0
    base_right = base_wrong = base_errors = 0
    cand_right = cand_wrong = cand_errors = 0
    raw_equal = decoded_equal = one_sided_errors = 0
    max_abs_difference = 0.0
    first_failure: dict[str, object] | None = None
    while valid < COUNT:
        attempts += 1
        try:
            example = generator.generate()
        except Exception as exc:  # noqa: BLE001 - errors are audit data
            generation_errors += 1
            if first_failure is None:
                first_failure = {"stage": "generate", "attempt": attempts, "error": repr(exc)}
            continue
        if max(len(example["input"]), len(example["input"][0])) > 30:
            oversize += 1
            continue
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        valid += 1
        base_raw = cand_raw = None
        base_error = cand_error = None
        try:
            base_raw = baseline.run(["output"], {"input": benchmark["input"]})[0]
        except Exception as exc:  # noqa: BLE001 - errors are audit data
            base_error = exc
            base_errors += 1
        try:
            cand_raw = candidate.run(["output"], {"input": benchmark["input"]})[0]
        except Exception as exc:  # noqa: BLE001 - errors are audit data
            cand_error = exc
            cand_errors += 1
        if (base_error is None) != (cand_error is None):
            one_sided_errors += 1
        expected = benchmark["output"] > 0.0
        if base_raw is not None:
            if np.array_equal(base_raw > 0.0, expected):
                base_right += 1
            else:
                base_wrong += 1
        if cand_raw is not None:
            if np.array_equal(cand_raw > 0.0, expected):
                cand_right += 1
            else:
                cand_wrong += 1
                if first_failure is None:
                    first_failure = {
                        "stage": "candidate_wrong",
                        "valid_case": valid,
                        "input_shape": list(np.asarray(example["input"]).shape),
                        "differing_elements": int(np.count_nonzero((cand_raw > 0.0) != expected)),
                    }
        elif first_failure is None:
            first_failure = {"stage": "candidate_runtime", "valid_case": valid, "error": repr(cand_error)}
        if base_raw is not None and cand_raw is not None:
            decoded_equal += int(np.array_equal(base_raw > 0.0, cand_raw > 0.0))
            if np.array_equal(base_raw, cand_raw):
                raw_equal += 1
            else:
                difference = np.abs(
                    np.nan_to_num(base_raw, nan=0.0, posinf=0.0, neginf=0.0)
                    - np.nan_to_num(cand_raw, nan=0.0, posinf=0.0, neginf=0.0)
                )
                max_abs_difference = max(max_abs_difference, float(difference.max()))
    return {
        "seed": seed,
        "mode": "ORT_DISABLE_ALL" if disable_all else "ORT_DEFAULT",
        "requested_valid": COUNT,
        "valid": valid,
        "attempts": attempts,
        "generation_errors": generation_errors,
        "oversize_skipped": oversize,
        "baseline": {"right": base_right, "wrong": base_wrong, "errors": base_errors},
        "candidate": {"right": cand_right, "wrong": cand_wrong, "errors": cand_errors},
        "differential": {
            "raw_bitwise_equal": raw_equal,
            "decoded_equal": decoded_equal,
            "one_sided_errors": one_sided_errors,
            "max_abs_raw_difference": max_abs_difference,
        },
        "first_failure": first_failure,
        "elapsed_seconds": time.monotonic() - start,
    }


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append("")
    return result


def structural(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    banned = []
    custom_domains = []
    nested = []
    negative_conv_pads = []
    max_einsum_inputs = 0
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in BANNED or "SEQUENCE" in upper:
            banned.append(node.op_type)
        if node.domain not in ("", "ai.onnx"):
            custom_domains.append(node.domain)
        if node.op_type == "Einsum":
            max_einsum_inputs = max(max_einsum_inputs, len(node.input))
            equation = next(
                helper.get_attribute_value(attr).decode("ascii")
                for attr in node.attribute
                if attr.name == "equation"
            )
            if len(equation.split("->")[0].split(",")) != len(node.input):
                raise RuntimeError("Einsum operand/equation mismatch")
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                nested.append(node.op_type)
        if node.op_type in ("Conv", "ConvTranspose", "QLinearConv"):
            pads = next(
                (helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "pads"),
                [],
            )
            if any(int(value) < 0 for value in pads):
                negative_conv_pads.append({"output": node.output[0], "pads": list(pads)})
    nonstatic = []
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if not value.type.HasField("tensor_type") or any(
            not dim.HasField("dim_value") or dim.dim_value <= 0
            for dim in value.type.tensor_type.shape.dim
        ):
            nonstatic.append(value.name)
    nonfinite = []
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            nonfinite.append(init.name)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "file_bytes": path.stat().st_size,
        "under_1_44mb": path.stat().st_size < 1_440_000,
        "checker_full": "pass",
        "strict_shape_inference": "pass",
        "standard_domains": not custom_domains,
        "custom_domains": custom_domains,
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": nested,
        "banned_or_sequence_ops": banned,
        "conv_bias_issues": check_conv_bias(model),
        "negative_conv_pads": negative_conv_pads,
        "dynamic_or_nonpositive_tensors": nonstatic,
        "nonfinite_initializers": nonfinite,
        "value_info_count": len(model.graph.value_info),
        "node_count": len(model.graph.node),
        "parameter_elements": sum(numpy_helper.to_array(item).size for item in model.graph.initializer),
        "max_einsum_inputs": max_einsum_inputs,
        "op_histogram": dict(sorted(collections.Counter(node.op_type for node in model.graph.node).items())),
        "input_shape": dims(inferred.graph.input[0]),
        "output_shape": dims(inferred.graph.output[0]),
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    model = onnx.load(CANDIDATE)
    profile = scoring.score_and_verify(
        copy.deepcopy(model), TASK, str(HERE / "score_work"), label="a10_048", require_correct=True
    )
    margin_ok, margin_min = scoring.model_margin_stable(copy.deepcopy(model), TASK)
    baseline_structure = structural(BASELINE)
    candidate_structure = structural(CANDIDATE)
    report = {
        "task": TASK,
        "baseline": baseline_structure,
        "candidate": candidate_structure,
        "comparison": {
            "node_count_delta": candidate_structure["node_count"] - baseline_structure["node_count"],
            "parameter_delta": candidate_structure["parameter_elements"]
            - baseline_structure["parameter_elements"],
            "value_info_delta": candidate_structure["value_info_count"]
            - baseline_structure["value_info_count"],
            "only_semantic_edit": "fold unit singleton-axis Einsum operand into shapes of existing coefficient operands",
        },
        "profile": profile,
        "projected_gain": math.log(379.0 / 378.0),
        "known_disable_all": known(make_session(CANDIDATE, True)),
        "known_default_ort": known(make_session(CANDIDATE, False)),
        "fresh_disable_all": fresh(True, 8_048_000_001),
        "fresh_default_ort": fresh(False, 8_048_000_002),
        "official_gold": official_gold(CANDIDATE, TASK),
        "margin": {"stable": bool(margin_ok), "minimum": margin_min},
    }
    (HERE / "task048_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
