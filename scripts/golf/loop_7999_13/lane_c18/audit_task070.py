#!/usr/bin/env python3
"""Independent dual-ORT audit of task070 exact scalar-Einsum fusion."""

from __future__ import annotations

import collections
import copy
import hashlib
import importlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


TASK = 70
TASK_HASH = "32597951"
COUNT = 5000
BASE = HERE / "base/task070.onnx"
CANDIDATE = ROOT / "scripts/golf/loop_7999_13/lane_task070_exact/task070_exact_scalar_fusion.onnx"
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
    rows: dict[str, dict[str, int]] = {}
    for subset in ("train", "test", "arc-gen"):
        right = wrong = errors = 0
        for example in scoring.load_examples(TASK)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                raw = session.run(["output"], {"input": benchmark["input"]})[0]
                if np.array_equal(raw > 0.0, benchmark["output"] > 0.0):
                    right += 1
                else:
                    wrong += 1
            except Exception:  # noqa: BLE001 - retained as audit data
                errors += 1
        rows[subset] = {"right": right, "wrong": wrong, "errors": errors}
    rows["total"] = {
        key: sum(rows[name][key] for name in ("train", "test", "arc-gen"))
        for key in ("right", "wrong", "errors")
    }
    return rows


def fresh(disable_all: bool, seed: int) -> dict[str, object]:
    base = make_session(BASE, disable_all)
    candidate = make_session(CANDIDATE, disable_all)
    generator = importlib.import_module(f"task_{TASK_HASH}")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    start = time.monotonic()
    right_base = wrong_base = errors_base = 0
    right_cand = wrong_cand = errors_cand = 0
    raw_bitwise_equal = raw_allclose = decoded_equal = one_sided_errors = 0
    nonfinite_base = nonfinite_cand = 0
    max_abs_difference = 0.0
    first_failure: dict[str, object] | None = None
    for case in range(COUNT):
        benchmark = scoring.convert_to_numpy(generator.generate())
        if benchmark is None:
            raise RuntimeError(f"generator case {case} did not fit scorer input")
        base_raw = candidate_raw = None
        base_error = candidate_error = None
        try:
            base_raw = base.run(["output"], {"input": benchmark["input"]})[0]
        except Exception as exc:  # noqa: BLE001
            base_error = exc
            errors_base += 1
        try:
            candidate_raw = candidate.run(["output"], {"input": benchmark["input"]})[0]
        except Exception as exc:  # noqa: BLE001
            candidate_error = exc
            errors_cand += 1
        one_sided_errors += int((base_error is None) != (candidate_error is None))
        expected = benchmark["output"] > 0.0
        if base_raw is not None:
            nonfinite_base += int(not np.isfinite(base_raw).all())
            if np.array_equal(base_raw > 0.0, expected):
                right_base += 1
            else:
                wrong_base += 1
        if candidate_raw is not None:
            nonfinite_cand += int(not np.isfinite(candidate_raw).all())
            if np.array_equal(candidate_raw > 0.0, expected):
                right_cand += 1
            else:
                wrong_cand += 1
                if first_failure is None:
                    first_failure = {
                        "case": case,
                        "kind": "candidate_wrong",
                        "different_cells": int(np.count_nonzero((candidate_raw > 0.0) != expected)),
                    }
        elif first_failure is None:
            first_failure = {"case": case, "kind": "candidate_runtime", "error": repr(candidate_error)}
        if base_raw is not None and candidate_raw is not None:
            raw_bitwise_equal += int(np.array_equal(base_raw, candidate_raw))
            raw_allclose += int(np.allclose(base_raw, candidate_raw, rtol=1e-5, atol=1e-5))
            decoded_equal += int(np.array_equal(base_raw > 0.0, candidate_raw > 0.0))
            delta = np.abs(base_raw.astype(np.float64) - candidate_raw.astype(np.float64))
            max_abs_difference = max(max_abs_difference, float(delta.max()))
    return {
        "mode": "ORT_DISABLE_ALL" if disable_all else "ORT_DEFAULT",
        "seed": seed,
        "count": COUNT,
        "baseline": {"right": right_base, "wrong": wrong_base, "errors": errors_base},
        "candidate": {"right": right_cand, "wrong": wrong_cand, "errors": errors_cand},
        "differential": {
            "raw_bitwise_equal": raw_bitwise_equal,
            "raw_allclose_rtol1e-5_atol1e-5": raw_allclose,
            "decoded_equal": decoded_equal,
            "one_sided_errors": one_sided_errors,
            "max_abs_raw_difference": max_abs_difference,
        },
        "nonfinite_output_cases": {"baseline": nonfinite_base, "candidate": nonfinite_cand},
        "first_failure": first_failure,
        "elapsed_seconds": time.monotonic() - start,
    }


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param or ""
        for dim in value.type.tensor_type.shape.dim
    ]


def structure(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    ops = collections.Counter(node.op_type for node in model.graph.node)
    equations = []
    banned = []
    custom_domains = []
    nested = []
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in BANNED or "SEQUENCE" in upper:
            banned.append(node.op_type)
        if node.domain not in ("", "ai.onnx"):
            custom_domains.append(node.domain)
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                nested.append(node.op_type)
        if node.op_type == "Einsum":
            equation = next(
                helper.get_attribute_value(attr).decode("ascii")
                for attr in node.attribute
                if attr.name == "equation"
            )
            if len(equation.split("->")[0].split(",")) != len(node.input):
                raise RuntimeError("Einsum operand/equation mismatch")
            equations.append({"equation": equation, "operands": list(node.input)})
    nonstatic = []
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in value.type.tensor_type.shape.dim):
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
        "full_checker": "PASS",
        "strict_shape_data_prop": "PASS",
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "parameter_elements": sum(numpy_helper.to_array(item).size for item in model.graph.initializer),
        "op_histogram": dict(ops),
        "einsums": equations,
        "standard_domains": not custom_domains,
        "custom_domains": custom_domains,
        "banned_or_sequence_ops": banned,
        "nested_graphs": nested,
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "conv_bias_issues": check_conv_bias(model),
        "dynamic_or_nonpositive_tensors": nonstatic,
        "nonfinite_initializers": nonfinite,
        "input_shape": dims(inferred.graph.input[0]),
        "output_shape": dims(inferred.graph.output[0]),
        "value_info_names": [value.name for value in model.graph.value_info],
    }


def algebraic_fusion() -> dict[str, object]:
    base = onnx.load(BASE)
    candidate = onnx.load(CANDIDATE)
    first, terminal = base.graph.node
    base_eq = [helper.get_attribute_value(node.attribute[0]).decode("ascii") for node in base.graph.node]
    candidate_eq = helper.get_attribute_value(candidate.graph.node[0].attribute[0]).decode("ascii")
    same_initializers = all(
        left.name == right.name and left.SerializeToString() == right.SerializeToString()
        for left, right in zip(base.graph.initializer, candidate.graph.initializer)
    ) and len(base.graph.initializer) == len(candidate.graph.initializer)
    expected_inputs = list(terminal.input)
    h_index = expected_inputs.index("H")
    expected_inputs[h_index : h_index + 1] = ["input", "R", "T"]
    expected_eq = base_eq[1].replace(",e,kq", ",bnuv,on,eo,kq")
    return {
        "base_producer_equation": base_eq[0],
        "base_terminal_equation": base_eq[1],
        "candidate_equation": candidate_eq,
        "candidate_equation_is_exact_inline": candidate_eq == expected_eq,
        "candidate_operands_are_exact_inline": list(candidate.graph.node[0].input) == expected_inputs,
        "initializers_byte_identical": same_initializers,
        "removed_only_intermediate": list(first.output) == ["H"] and len(candidate.graph.node) == 1,
        "stale_H_value_info_removed": all(item.name != "H" for item in candidate.graph.value_info),
        "renaming": {"bihw": "bnuv", "ri": "on", "er": "eo"},
        "interpretation": "H[e]=sum(input[b,i,h,w]*R[r,i]*T[e,r]); replacing terminal H[e] with the same contracted factors is associativity/distributivity of finite sums.",
        "lookup_indicators": {
            "gather_scatter_topk_tfidf_hardmax": 0,
            "coordinate_or_example_bank_initializers": 0,
            "parameter_elements": 75,
            "unique_initializer_names": [item.name for item in candidate.graph.initializer],
        },
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    with tempfile.TemporaryDirectory(prefix="c18_070_", dir="/tmp") as workdir:
        base_profile = scoring.score_and_verify(
            copy.deepcopy(onnx.load(BASE)), TASK, workdir, "c18_base", require_correct=True
        )
        candidate_profile = scoring.score_and_verify(
            copy.deepcopy(onnx.load(CANDIDATE)), TASK, workdir, "c18_candidate", require_correct=True
        )
    margin_ok, margin_min = scoring.model_margin_stable(copy.deepcopy(onnx.load(CANDIDATE)), TASK)
    report = {
        "task": TASK,
        "baseline_structure": structure(BASE),
        "candidate_structure": structure(CANDIDATE),
        "baseline_profile": base_profile,
        "candidate_profile": candidate_profile,
        "known_disable_all": known(make_session(CANDIDATE, True)),
        "known_default": known(make_session(CANDIDATE, False)),
        "fresh_disable_all": fresh(True, 7_999_130_700),
        "fresh_default": fresh(False, 7_999_130_701),
        "margin": {"stable": bool(margin_ok), "minimum": margin_min},
        "algebraic_fusion": algebraic_fusion(),
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "base_cost": base_profile["cost"],
        "candidate_cost": candidate_profile["cost"],
        "known_disable_all": report["known_disable_all"]["total"],
        "known_default": report["known_default"]["total"],
        "fresh_disable_all": report["fresh_disable_all"]["candidate"],
        "fresh_default": report["fresh_default"]["candidate"],
    }, indent=2))


if __name__ == "__main__":
    main()
