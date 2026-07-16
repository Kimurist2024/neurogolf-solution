#!/usr/bin/env python3
"""Profile exact-local probes and expose the task008 authority's runtime shapes."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "inputs/arc-gen-repo/tasks")]
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


MODELS = {
    "authority": HERE / "authority/task008.onnx",
    "derive_two": HERE / "probes/task008_derive_two.onnx",
    "derive_three": HERE / "probes/task008_derive_three.onnx",
    "derive_five": HERE / "probes/task008_derive_five.onnx",
}
MODES = (
    (True, 1, "disable_all_t1"),
    (True, 4, "disable_all_t4"),
    (False, 1, "default_t1"),
    (False, 4, "default_t4"),
)
FRESH_SEED = 194_008
FRESH_COUNT = 2000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_session(model: onnx.ModelProto, disable: bool, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def evaluate(session: ort.InferenceSession, examples: list[dict[str, Any]]) -> dict[str, Any]:
    right = wrong = errors = nonfinite = 0
    shapes: set[tuple[int, ...]] = set()
    first_failure = None
    for index, example in enumerate(examples):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        try:
            raw = np.asarray(
                session.run(
                    [session.get_outputs()[0].name],
                    {session.get_inputs()[0].name: benchmark["input"]},
                )[0]
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            first_failure = first_failure or {
                "case": index,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        shapes.add(tuple(int(value) for value in raw.shape))
        nonfinite += int(raw.size - np.count_nonzero(np.isfinite(raw)))
        if np.array_equal(raw > 0, benchmark["output"] > 0):
            right += 1
        else:
            wrong += 1
            first_failure = first_failure or {"case": index, "shape": list(raw.shape)}
    declared = session.get_outputs()[0].shape
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "nonfinite_output_values": nonfinite,
        "declared_output_shape": declared,
        "runtime_output_shapes": [list(shape) for shape in sorted(shapes)],
        "declared_runtime_match": all(list(shape) == declared for shape in shapes),
        "first_failure": first_failure,
    }


def unknown_like(value: onnx.ValueInfoProto) -> onnx.ValueInfoProto:
    tensor = value.type.tensor_type
    rank = len(tensor.shape.dim)
    return helper.make_tensor_value_info(value.name, tensor.elem_type, [None] * rank)


def trace_all(model: onnx.ModelProto, examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Remove profiling annotations and expose every node output with unknown dims."""
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    infos = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    traced.graph.ClearField("value_info")
    traced.graph.ClearField("output")
    seen: set[str] = set()
    for node in traced.graph.node:
        for name in node.output:
            if name and name not in seen:
                traced.graph.output.append(unknown_like(infos[name]))
                seen.add(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    output_names = [value.name for value in session.get_outputs()]
    shapes: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    max_bytes: dict[str, int] = defaultdict(int)
    nonfinite: dict[str, int] = defaultdict(int)
    for example in examples:
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        values = session.run(None, {session.get_inputs()[0].name: benchmark["input"]})
        for name, value in zip(output_names, values):
            array = np.asarray(value)
            shapes[name].add(tuple(int(dim) for dim in array.shape))
            max_bytes[name] = max(max_bytes[name], int(array.nbytes))
            if np.issubdtype(array.dtype, np.number):
                nonfinite[name] += int(array.size - np.count_nonzero(np.isfinite(array)))
    rows = {
        name: {
            "runtime_shapes": [list(shape) for shape in sorted(shapes[name])],
            "max_runtime_bytes": max_bytes[name],
            "nonfinite_values": nonfinite[name],
        }
        for name in output_names
    }
    return {
        "cases": len(examples),
        "runtime_tensors": len(output_names),
        "total_nonfinite_values": sum(nonfinite.values()),
        "max_sum_runtime_bytes": sum(max_bytes.values()),
        "selected": {
            name: rows[name]
            for name in (
                "input_cloak",
                "input_f16_hidden",
                "input_i8_hidden",
                "red_crop_i8_4",
                "grid3_f16",
                "indices2x2xhwx5_f16",
                "indices_cloak_f16",
                "indices_i64_hidden",
                "updates2x2hw_i8",
                "output",
            )
        },
        "all": rows,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    known = [
        example
        for group in scoring.load_examples(8).values()
        for example in group
    ]
    generator = importlib.import_module("task_05f2a901")
    random.seed(FRESH_SEED)
    fresh = [generator.generate() for _ in range(FRESH_COUNT)]
    rows = []
    for label, path in MODELS.items():
        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        nonstatic = [
            value.name
            for value in list(inferred.graph.value_info) + list(inferred.graph.output)
            if any(not dim.HasField("dim_value") or dim.dim_value <= 0
                   for dim in value.type.tensor_type.shape.dim)
        ]
        with tempfile.TemporaryDirectory(prefix=f"task008_{label}_", dir="/tmp") as workdir:
            profile = scoring.score_and_verify(
                copy.deepcopy(model), 8, workdir, label, require_correct=False
            )
        configs = {
            mode: evaluate(make_session(model, disable, threads), known)
            for disable, threads, mode in MODES
        }
        row = {
            "label": label,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "checker_full": True,
            "strict_data_prop": True,
            "nonstatic_inferred": nonstatic,
            "standard_domains": all(
                node.domain in ("", "ai.onnx") for node in model.graph.node
            ) and all(item.domain in ("", "ai.onnx") for item in model.opset_import),
            "conv_bias_findings": check_conv_bias(model),
            "profile": profile,
            "known_four_configs": configs,
            "strict_lower": bool(profile and profile["cost"] < 431),
        }
        rows.append(row)
        print(label, profile, "strict_lower", row["strict_lower"], flush=True)

    authority = onnx.load(MODELS["authority"])
    fresh_configs = {
        mode: evaluate(make_session(authority, disable, threads), fresh)
        for disable, threads, mode in MODES
    }
    trace = trace_all(authority, fresh[:32])
    result = {
        "task": 8,
        "generator": "inputs/arc-gen-repo/tasks/task_05f2a901.py",
        "fresh_seed": FRESH_SEED,
        "fresh_count": FRESH_COUNT,
        "rows": rows,
        "authority_fresh_four_configs": fresh_configs,
        "authority_all_intermediate_trace": trace,
        "accepted": [],
        "decision": "NO_STRICT_LOWER_TRUTHFUL_CANDIDATE",
    }
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("trace selected", json.dumps(trace["selected"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
