#!/usr/bin/env python3
"""Cost, correctness, static-shape, operation, and Conv-bias audit."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


BANNED = {
    "Loop",
    "Scan",
    "NonZero",
    "Unique",
    "Script",
    "Function",
    "Compress",
    "TfIdfVectorizer",
    "Hardmax",
}


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str | None]:
    shape = value.type.tensor_type.shape
    result: list[int | str | None] = []
    for dim in shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def audit(task: int, label: str, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    model = onnx.load_from_string(payload)
    checker_ok = data_prop_ok = static_positive = False
    checker_error = data_prop_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
        checker_ok = True
    except Exception as exc:  # noqa: BLE001
        checker_error = repr(exc)
    inferred = None
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model,
            strict_mode=True,
            data_prop=True,
        )
        data_prop_ok = True
        static_positive = True
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
            if not value.type.HasField("tensor_type"):
                static_positive = False
                continue
            for dim in value.type.tensor_type.shape.dim:
                if not dim.HasField("dim_value") or dim.dim_value <= 0:
                    static_positive = False
    except Exception as exc:  # noqa: BLE001
        data_prop_error = repr(exc)

    ops = collections.Counter(node.op_type for node in model.graph.node)
    banned = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type in BANNED or "Sequence" in node.op_type
        }
    )
    nested = sum(
        attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attribute in node.attribute
    )
    einsum_inputs = [len(node.input) for node in model.graph.node if node.op_type == "Einsum"]
    giant_einsum = any(count > 3 for count in einsum_inputs)

    with tempfile.TemporaryDirectory() as workdir:
        score = scoring.score_and_verify(
            model,
            task,
            workdir,
            label=f"{label}_{task}",
            require_correct=True,
        )
    session = scoring._make_raw_session(model)
    examples = scoring.load_examples(task)
    known = examples["train"] + examples["test"] + examples["arc-gen"]
    known_right = known_wrong = -1
    if session is not None:
        known_right, known_wrong, _ = scoring.verify_subset(session, known)

    runtime: dict[str, object] = {}
    sanitized = scoring.sanitize_model(model)
    for disabled, runtime_label in ((True, "disable_all"), (False, "default")):
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            if disabled:
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            runtime_session = ort.InferenceSession(sanitized.SerializeToString(), options)
            benchmark = scoring.convert_to_numpy(known[0])
            raw = scoring._raw_output(runtime_session, benchmark["input"])
            runtime[runtime_label] = {
                "ok": True,
                "output_shape": list(raw.shape),
            }
        except Exception as exc:  # noqa: BLE001
            runtime[runtime_label] = {"ok": False, "error": repr(exc)}

    row = {
        "task": task,
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "serialized_bytes": len(payload),
        "checker_full": checker_ok,
        "checker_error": checker_error,
        "strict_shape_data_prop": data_prop_ok,
        "data_prop_error": data_prop_error,
        "all_inferred_dims_static_positive": static_positive,
        "input_shapes": [tensor_shape(value) for value in model.graph.input],
        "output_shapes": [tensor_shape(value) for value in model.graph.output],
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": nested,
        "banned_ops": banned,
        "einsum_input_counts": einsum_inputs,
        "giant_einsum": giant_einsum,
        "conv_bias_issues": [list(issue) for issue in check_conv_bias(model)],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_histogram": dict(sorted(ops.items())),
        "known_right": known_right,
        "known_wrong_or_errors": known_wrong,
        "cost": score["cost"] if score else None,
        "memory": score["memory"] if score else None,
        "params": score["params"] if score else None,
        "task_score": (
            max(1.0, 25.0 - math.log(score["cost"]))
            if score and score.get("cost")
            else None
        ),
        "runtime_probe": runtime,
    }
    return row


def main() -> None:
    jobs = []
    for task in (168, 192, 343, 344):
        jobs.append((task, "baseline_8003_40", HERE / "baseline" / f"task{task:03d}.onnx"))
        jobs.append((task, "true_rule_control", HERE / "models" / f"task{task:03d}.onnx"))
    jobs.append((344, "wave1_policy95", HERE / "wave1" / "task344.onnx"))
    rows = [audit(*job) for job in jobs]
    for row in rows:
        print(json.dumps(row), flush=True)
    (HERE / "model_audit.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
