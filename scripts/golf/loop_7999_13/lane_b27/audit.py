#!/usr/bin/env python3
"""Strict non-promoting audit of task382 shape-repair probes."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402

from build_candidates import trace_shapes  # noqa: E402


ITEMS = {
    "baseline": HERE / "baseline_task382.onnx",
    "source_spoofed": ROOT / "scripts/golf/loop_7999_13/lane_headroom/candidates/task382.onnx",
    "output_shape_only": HERE / "task382_output_shape_only.onnx",
    "declared_horizontal_runtime": HERE / "task382_declared_horizontal_runtime.onnx",
    "declared_vertical_runtime": HERE / "task382_declared_vertical_runtime.onnx",
}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dimension.dim_value)
        if dimension.HasField("dim_value")
        else dimension.dim_param or "?"
        for dimension in value.type.tensor_type.shape.dim
    ]


def structural(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    checker_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        checker_full = strict_inference = True
    except Exception as exc:
        checker_error = f"{type(exc).__name__}: {exc}"
        checker_full = strict_inference = False
        inferred = model
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    static_positive = all(
        value.type.HasField("tensor_type")
        and all(
            dimension.HasField("dim_value") and dimension.dim_value > 0
            for dimension in value.type.tensor_type.shape.dim
        )
        for value in values
    )
    try:
        memory, params, cost = (int(value) for value in cost_of(str(path)))
        cost_error = None
    except Exception as exc:
        memory = params = cost = None
        cost_error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "serialized_bytes": path.stat().st_size,
        "checker_full": checker_full,
        "strict_shape_inference": strict_inference,
        "checker_error": checker_error,
        "static_positive_shapes": static_positive,
        "declared_output_shape": shape(model.graph.output[0]),
        "standard_domains": all(item.domain in ("", "ai.onnx") for item in model.opset_import)
        and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
        "no_banned_or_nested": not model.functions
        and not model.graph.sparse_initializer
        and all(
            node.op_type.upper() not in BANNED
            and "SEQUENCE" not in node.op_type.upper()
            and all(
                attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for attr in node.attribute
            )
            for node in model.graph.node
        ),
        "conv_bias_findings": check_conv_bias(model),
        "memory": memory,
        "params": params,
        "cost": cost,
        "cost_error": cost_error,
    }


def declared_runtime_mismatches(
    model: onnx.ModelProto, runtime_shapes: dict[str, tuple[int, ...]]
) -> list[dict[str, Any]]:
    declarations = {
        value.name: shape(value)
        for value in list(model.graph.value_info) + list(model.graph.output)
    }
    return [
        {"tensor": name, "declared": declared, "actual": list(runtime_shapes[name])}
        for name, declared in declarations.items()
        if name in runtime_shapes and declared != list(runtime_shapes[name])
    ]


def make_session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def known_dual(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    examples = scoring.load_examples(382)
    result: dict[str, Any] = {}
    for label, disabled in (("disable_all", True), ("default", False)):
        right = wrong = errors = 0
        minimum_positive: float | None = None
        first_failure = None
        try:
            session = make_session(model, disabled)
        except Exception as exc:
            result[label] = {
                "right": 0,
                "wrong": 0,
                "errors": 1,
                "session_error": f"{type(exc).__name__}: {exc}",
            }
            continue
        for subset in ("train", "test", "arc-gen"):
            for index, example in enumerate(examples[subset]):
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    continue
                try:
                    raw = np.asarray(
                        session.run(["output"], {"input": benchmark["input"]})[0]
                    )
                    positive = raw[raw > 0]
                    if positive.size:
                        value = float(positive.min())
                        minimum_positive = value if minimum_positive is None else min(minimum_positive, value)
                    if np.array_equal(raw > 0, benchmark["output"].astype(bool)):
                        right += 1
                    else:
                        wrong += 1
                        first_failure = first_failure or {"subset": subset, "index": index, "kind": "wrong"}
                except Exception as exc:
                    errors += 1
                    first_failure = first_failure or {
                        "subset": subset,
                        "index": index,
                        "kind": "runtime_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        result[label] = {
            "right": right,
            "wrong": wrong,
            "errors": errors,
            "min_positive": minimum_positive,
            "first_failure": first_failure,
        }
    return result


def main() -> int:
    examples = scoring.load_examples(382)["train"]
    horizontal_input = scoring.convert_to_numpy(examples[0])["input"]
    vertical_input = scoring.convert_to_numpy(examples[2])["input"]
    source = onnx.load(ITEMS["source_spoofed"])
    horizontal_shapes = trace_shapes(source, horizontal_input)
    vertical_shapes = trace_shapes(source, vertical_input)

    structures = {name: structural(path) for name, path in ITEMS.items()}
    runtime_truth = {}
    for name, path in ITEMS.items():
        model = onnx.load(path)
        runtime_truth[name] = {
            "horizontal_mismatches": declared_runtime_mismatches(model, horizontal_shapes),
            "vertical_mismatches": declared_runtime_mismatches(model, vertical_shapes),
        }

    known = {
        name: known_dual(ITEMS[name])
        for name in ("output_shape_only", "declared_horizontal_runtime")
    }
    payload = {
        "task": 382,
        "baseline_score_label": 8000.46,
        "baseline_cost": 820,
        "source_claimed_cost": 814,
        "structures": structures,
        "runtime_truth": runtime_truth,
        "known_dual": known,
        "decision": {
            "status": "REJECT_COST_AND_SHAPE",
            "winner_count": 0,
            "reasons": [
                "output-only repair fails full ONNX checker and still has allocator failures in default ORT",
                "declaring observed intermediate shapes raises cost from 814 to 55417, above baseline 820",
                "eight intermediates swap row/column shapes by gravity orientation, so a single static declaration copied from either orientation is not truthful for the other",
                "fresh5000 and external acceptance validation are intentionally skipped because the same-or-cheaper prerequisite failed",
            ],
        },
    }
    (HERE / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["decision"], indent=2))
    print(json.dumps({name: {key: row[key] for key in ("checker_full", "declared_output_shape", "cost", "conv_bias_findings")} for name, row in structures.items()}, indent=2))
    print(json.dumps(known, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
