#!/usr/bin/env python3
"""Dual-ORT known and static audit of all eight pinned baseline members."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "golf"))
from lib import scoring  # noqa: E402
import check_conv_bias  # noqa: E402


TASKS = [37, 92, 132, 159, 218, 226, 228, 297]
ZIPS = [ROOT / "submission_base_8004.50.zip", ROOT / "submission_base_8005.16.zip"]
OUT = HERE / "baseline_known_static.json"
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def session(model: onnx.ModelProto, disable_all: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def main() -> None:
    ort.set_default_logger_severity(4)
    rows = []
    for task in TASKS:
        members = []
        data = b""
        for path in ZIPS:
            with zipfile.ZipFile(path) as archive:
                data = archive.read(f"task{task:03d}.onnx")
            members.append(
                {
                    "zip": path.name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        bad_shapes = []
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(
            inferred.graph.output
        ):
            if not value.type.HasField("tensor_type"):
                continue
            tensor_type = value.type.tensor_type
            if not tensor_type.HasField("shape") or any(
                not dim.HasField("dim_value")
                or dim.HasField("dim_param")
                or dim.dim_value <= 0
                for dim in tensor_type.shape.dim
            ):
                bad_shapes.append(value.name)

        examples = []
        loaded = scoring.load_examples(task)
        for subset in ("train", "test", "arc-gen"):
            examples.extend(loaded.get(subset, []))
        known = {}
        for disabled, label in ((True, "disable_all"), (False, "default")):
            right = errors = 0
            try:
                run = session(model, disabled)
                for example in examples:
                    converted = scoring.convert_to_numpy(example)
                    if converted is None:
                        continue
                    try:
                        raw = run.run(["output"], {"input": converted["input"]})[0]
                        right += int(np.array_equal(raw > 0, converted["output"] > 0))
                    except Exception:  # noqa: BLE001
                        errors += 1
            except Exception:  # noqa: BLE001
                errors = len(examples)
            first100_right = first100_errors = 0
            try:
                run100 = session(model, disabled)
                for example in examples[:100]:
                    converted = scoring.convert_to_numpy(example)
                    assert converted is not None
                    try:
                        raw = run100.run(["output"], {"input": converted["input"]})[0]
                        first100_right += int(
                            np.array_equal(raw > 0, converted["output"] > 0)
                        )
                    except Exception:  # noqa: BLE001
                        first100_errors += 1
            except Exception:  # noqa: BLE001
                first100_errors = min(100, len(examples))
            known[label] = {
                "first100_right": first100_right,
                "first100_total": min(100, len(examples)),
                "first100_runtime_errors": first100_errors,
                "all_right": right,
                "all_total": len(examples),
                "all_runtime_errors": errors,
            }

        max_einsum_inputs = max(
            [len(node.input) for node in model.graph.node if node.op_type == "Einsum"] or [0]
        )
        rows.append(
            {
                "task": task,
                "members": members,
                "rebase_compatible": len({item["sha256"] for item in members}) == 1,
                "checker_full": "PASS",
                "strict_shape_inference_data_prop": "PASS",
                "static_shape_failures": bad_shapes,
                "standard_domains": all(
                    item.domain in {"", "ai.onnx"} for item in model.opset_import
                ),
                "banned_ops": [
                    node.op_type
                    for node in model.graph.node
                    if node.op_type.upper() in BANNED
                    or "SEQUENCE" in node.op_type.upper()
                ],
                "nested_graphs": [
                    node.op_type
                    for node in model.graph.node
                    for attr in node.attribute
                    if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                ],
                "functions": len(model.functions),
                "conv_bias_ub": [list(item) for item in check_conv_bias.check_model(model)],
                "nodes": len(model.graph.node),
                "op_histogram": dict(Counter(node.op_type for node in model.graph.node)),
                "max_einsum_inputs": max_einsum_inputs,
                "known": known,
            }
        )
    result = {"baseline": ZIPS[-1].name, "tasks": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
