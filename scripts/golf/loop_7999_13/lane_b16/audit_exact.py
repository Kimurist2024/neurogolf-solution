#!/usr/bin/env python3
"""Recompute B16 baseline costs and reject structural/private-test hazards."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


TASKS = (157, 319)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value)
        if dim.HasField("dim_value")
        else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def static_positive(value: onnx.ValueInfoProto) -> bool:
    return all(
        dim.HasField("dim_value") and int(dim.dim_value) > 0
        for dim in value.type.tensor_type.shape.dim
    )


def structure(model: onnx.ModelProto) -> dict[str, Any]:
    error = None
    try:
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        checker = strict = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        checker = strict = False
        error = f"{type(exc).__name__}: {exc}"
    values = (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    )
    checks = {
        "checker_full": checker,
        "strict_data_prop": strict,
        "canonical_io": (
            len(model.graph.input) == 1
            and len(model.graph.output) == 1
            and model.graph.input[0].name == "input"
            and model.graph.output[0].name == "output"
        ),
        "standard_domains": all(
            item.domain in ("", "ai.onnx") for item in model.opset_import
        )
        and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
        "no_functions_sparse_nested": (
            not model.functions
            and not model.graph.sparse_initializer
            and all(
                attribute.type
                not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for node in model.graph.node
                for attribute in node.attribute
            )
        ),
        "no_banned_ops": all(
            node.op_type.upper() not in BANNED
            and "SEQUENCE" not in node.op_type.upper()
            for node in model.graph.node
        ),
        "einsum_inputs_lt15": all(
            node.op_type != "Einsum" or len(node.input) < 15
            for node in model.graph.node
        ),
        "static_positive_shapes": all(static_positive(value) for value in values),
        "no_external_initializers": all(
            item.data_location != onnx.TensorProto.EXTERNAL
            and not item.external_data
            for item in model.graph.initializer
        ),
        "finite_initializers": all(
            array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
            for array in (
                numpy_helper.to_array(item) for item in model.graph.initializer
            )
        ),
        "conv_bias_safe": not check_conv_bias(model),
    }
    return {"checks": checks, "pass": all(checks.values()), "error": error}


def runtime_shapes(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=False, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    declared = {
        value.name: shape(value)
        for value in list(model.graph.value_info) + list(model.graph.output)
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
    session = ort.InferenceSession(traced.SerializeToString(), options)
    sample = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
    assert sample is not None
    outputs = session.run(names, {"input": sample["input"]})
    actual = {name: list(np.asarray(value).shape) for name, value in zip(names, outputs)}
    mismatches = [
        {"tensor": name, "declared": declared_shape, "actual": actual[name]}
        for name, declared_shape in declared.items()
        if name in actual and declared_shape != actual[name]
    ]
    return {"shape_cloak": bool(mismatches), "mismatches": mismatches}


def known_dual(model: onnx.ModelProto, task: int) -> list[dict[str, Any]]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return [{"mode": "session", "right": 0, "wrong": 0, "errors": 1}]
    rows: list[dict[str, Any]] = []
    examples = scoring.load_examples(task)
    for mode in ("disabled", "default"):
        options = ort.SessionOptions()
        if mode == "disabled":
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        right = wrong = errors = 0
        first_failure = None
        try:
            session = ort.InferenceSession(sanitized.SerializeToString(), options)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {"mode": mode, "right": 0, "wrong": 0, "errors": 1, "error": repr(exc)}
            )
            continue
        for subset in ("train", "test", "arc-gen"):
            for index, example in enumerate(examples[subset]):
                sample = scoring.convert_to_numpy(example)
                if sample is None:
                    continue
                try:
                    raw = session.run(["output"], {"input": sample["input"]})[0]
                    if np.array_equal(raw > 0.0, sample["output"].astype(bool)):
                        right += 1
                    else:
                        wrong += 1
                        first_failure = first_failure or {"subset": subset, "index": index}
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    first_failure = first_failure or {
                        "subset": subset,
                        "index": index,
                        "error": repr(exc),
                    }
        rows.append(
            {
                "mode": mode,
                "right": right,
                "wrong": wrong,
                "errors": errors,
                "first_failure": first_failure,
            }
        )
    return rows


def main() -> int:
    ort.set_default_logger_severity(4)
    rows: dict[str, Any] = {}
    for task in TASKS:
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        model = onnx.load(path)
        with tempfile.TemporaryDirectory(prefix=f"b16_{task:03d}_", dir="/tmp") as workdir:
            score = scoring.score_and_verify(
                copy.deepcopy(model), task, workdir, label="exact", require_correct=False
            )
        try:
            trace = runtime_shapes(model, task)
        except Exception as exc:  # noqa: BLE001
            trace = {"shape_cloak": None, "trace_error": f"{type(exc).__name__}: {exc}"}
        rows[str(task)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "actual_score": score,
            "structure": structure(model),
            "runtime_shapes": trace,
            "known_dual": known_dual(model, task),
        }
        print(task, score, trace, flush=True)
    (HERE / "exact_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
