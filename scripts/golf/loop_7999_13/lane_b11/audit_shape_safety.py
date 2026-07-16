#!/usr/bin/env python3
"""Audit exact B11 members for runtime shape truth and one-case dual-ORT safety."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from audit_candidates import conv_bias_findings, runtime_shape_trace  # noqa: E402
from lib import scoring  # noqa: E402


TASKS = (264, 281, 300, 358, 376, 387, 392)


def one_case(task: int, model: onnx.ModelProto, disable_all: bool) -> dict[str, object]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        assert sanitized is not None
        session = ort.InferenceSession(
            sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        case = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
        assert case is not None
        actual = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: case["input"]},
        )[0]
        return {
            "session": True,
            "correct": bool(np.array_equal(actual > 0, case["output"] > 0)),
            "output_shape": list(actual.shape),
        }
    except Exception as exc:  # noqa: BLE001
        return {"session": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ort.set_default_logger_severity(4)
    output: dict[str, object] = {}
    out_path = HERE / "baseline_shape_safety.json"
    for task in TASKS:
        model = onnx.load(HERE / "baseline" / f"task{task:03d}.onnx")
        row: dict[str, object] = {
            "task": task,
            "nodes": len(model.graph.node),
            "declared_intermediates": len(model.graph.value_info),
            "conv_bias_findings": conv_bias_findings(model),
        }
        try:
            onnx.checker.check_model(model, full_check=True)
            row["checker_full"] = True
        except Exception as exc:  # noqa: BLE001
            row.update(checker_full=False, checker_error=f"{type(exc).__name__}: {exc}")
        try:
            onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
            row["strict_shape_inference"] = True
        except Exception as exc:  # noqa: BLE001
            row.update(
                strict_shape_inference=False,
                strict_shape_error=f"{type(exc).__name__}: {exc}",
            )
        try:
            row["runtime_shape_trace"] = runtime_shape_trace(task, model)
        except Exception as exc:  # noqa: BLE001
            row["runtime_shape_trace"] = {"error": f"{type(exc).__name__}: {exc}"}
        row["disable_all_one_case"] = one_case(task, model, True)
        row["default_one_case"] = one_case(task, model, False)
        output[str(task)] = row
        out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        trace = row["runtime_shape_trace"]
        print(
            task,
            len(trace.get("declared_actual_mismatches", [])),
            row["disable_all_one_case"],
            row["default_one_case"],
            flush=True,
        )


if __name__ == "__main__":
    main()
