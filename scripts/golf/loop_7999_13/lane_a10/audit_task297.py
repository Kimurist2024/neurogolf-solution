#!/usr/bin/env python3
"""Dual-ORT, fresh-5000, differential, and structural audit for task297."""

from __future__ import annotations

import collections
import copy
import hashlib
import importlib
import json
import random
import sys
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


TASK = 297
TASK_HASH = "bd4472b8"
COUNT = 5000
BASELINE = HERE / "baseline" / "task297.onnx"
CANDIDATE = HERE / "task297_trim_zero_kernel.onnx"
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
    subsets: dict[str, dict[str, int]] = {}
    examples = scoring.load_examples(TASK)
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
            except Exception:  # noqa: BLE001 - errors are durable audit data
                errors += 1
        subsets[name] = {"right": right, "wrong": wrong, "errors": errors}
    subsets["total"] = {
        key: sum(int(subsets[name][key]) for name in ("train", "test", "arc-gen"))
        for key in ("right", "wrong", "errors")
    }
    return subsets


def fresh_and_differential(disable_all: bool) -> dict[str, object]:
    baseline = make_session(BASELINE, disable_all)
    candidate = make_session(CANDIDATE, disable_all)
    generator = importlib.import_module(f"task_{TASK_HASH}")
    seed = 7_999_130_000 + 10_000 * TASK + (0 if disable_all else 1)
    random.seed(seed)
    attempts = candidate_right = candidate_wrong = candidate_errors = 0
    baseline_right = baseline_wrong = baseline_errors = 0
    raw_equal = threshold_equal = one_sided_errors = 0
    first_failure: dict[str, object] | None = None
    while attempts < COUNT:
        example = generator.generate()
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        attempts += 1
        base_raw = cand_raw = None
        base_error = cand_error = None
        try:
            base_raw = baseline.run(["output"], {"input": benchmark["input"]})[0]
        except Exception as exc:  # noqa: BLE001 - errors are durable audit data
            base_error = exc
            baseline_errors += 1
        try:
            cand_raw = candidate.run(["output"], {"input": benchmark["input"]})[0]
        except Exception as exc:  # noqa: BLE001 - errors are durable audit data
            cand_error = exc
            candidate_errors += 1
        if (base_error is None) != (cand_error is None):
            one_sided_errors += 1
        expected = benchmark["output"] > 0.0
        if base_raw is not None:
            if np.array_equal(base_raw > 0.0, expected):
                baseline_right += 1
            else:
                baseline_wrong += 1
        if cand_raw is not None:
            if np.array_equal(cand_raw > 0.0, expected):
                candidate_right += 1
            else:
                candidate_wrong += 1
                if first_failure is None:
                    first_failure = {
                        "index": attempts - 1,
                        "kind": "wrong",
                        "input_shape": list(np.asarray(example["input"]).shape),
                        "differing_elements": int(np.count_nonzero((cand_raw > 0.0) != expected)),
                    }
        elif first_failure is None:
            first_failure = {
                "index": attempts - 1,
                "kind": "runtime",
                "error": repr(cand_error),
            }
        if base_raw is not None and cand_raw is not None:
            raw_equal += int(np.array_equal(base_raw, cand_raw))
            threshold_equal += int(np.array_equal(base_raw > 0.0, cand_raw > 0.0))
    return {
        "seed": seed,
        "attempts": attempts,
        "candidate": {
            "right": candidate_right,
            "wrong": candidate_wrong,
            "errors": candidate_errors,
        },
        "baseline": {
            "right": baseline_right,
            "wrong": baseline_wrong,
            "errors": baseline_errors,
        },
        "differential": {
            "raw_equal": raw_equal,
            "threshold_equal": threshold_equal,
            "one_sided_errors": one_sided_errors,
        },
        "first_candidate_failure": first_failure,
        "ort_disable_all": disable_all,
    }


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    output: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            output.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            output.append(dim.dim_param)
        else:
            output.append("")
    return output


def structural(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    banned: list[str] = []
    custom_domains: list[str] = []
    nested: list[str] = []
    negative_conv_pads: list[dict[str, object]] = []
    for node in model.graph.node:
        if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper():
            banned.append(node.op_type)
        if node.domain not in ("", "ai.onnx"):
            custom_domains.append(node.domain)
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
        "op_histogram": dict(sorted(collections.Counter(node.op_type for node in model.graph.node).items())),
        "input_shape": dims(inferred.graph.input[0]),
        "output_shape": dims(inferred.graph.output[0]),
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    model = onnx.load(CANDIDATE)
    profile = scoring.score_and_verify(
        copy.deepcopy(model), TASK, str(HERE / "score_work"), label="a10_297", require_correct=True
    )
    margin_ok, margin_min = scoring.model_margin_stable(copy.deepcopy(model), TASK)
    candidate_structure = structural(CANDIDATE)
    negative_pads = candidate_structure["negative_conv_pads"]
    report = {
        "task": TASK,
        "baseline": structural(BASELINE),
        "candidate": candidate_structure,
        "profile": profile,
        "known_disable_all": known(make_session(CANDIDATE, True)),
        "known_default_ort": known(make_session(CANDIDATE, False)),
        "fresh_disable_all": (
            fresh_and_differential(True)
            if not negative_pads
            else {"run": False, "reason": "negative Conv pads fail the structural gate"}
        ),
        "fresh_default_ort": (
            fresh_and_differential(False)
            if not negative_pads
            else {"run": False, "reason": "negative Conv pads fail the structural gate"}
        ),
        "margin": {"stable": bool(margin_ok), "minimum": margin_min},
        "onnx_schema_caveat": "Conv pads schema requires values >=0; candidate uses -24 end padding and is quarantined unless explicitly accepted as deterministic ORT crop behavior.",
    }
    (HERE / "task297_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
