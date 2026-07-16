#!/usr/bin/env python3
"""Rebase the repository-wide SHA history scanner on LB8009.46 for this lane."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py"
SPEC = importlib.util.spec_from_file_location("tasks182_204_208_284_history", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load exhaustive scanner")
SCANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCANNER)

SCANNER.HERE = HERE
SCANNER.TARGETS = (182, 204, 208, 284)
SCANNER.BASE_ZIP = ROOT / "submission_base_8009.46.zip"
SCANNER.CURRENT_COSTS_JSON = HERE / "authority_costs.json"

_strict_extra = SCANNER.strict_extra


def strict_extra_schema(data: bytes, sources: list[str]):
    passed, reasons, detail = _strict_extra(data, sources)
    model = onnx.load_model_from_string(data)
    nonfinite = []
    negative_conv_pads = []
    banned = []
    banned_names = {"Loop", "Scan", "NonZero", "Unique", "Compress"}
    for node in model.graph.node:
        if node.op_type in banned_names or "Sequence" in node.op_type:
            banned.append(node.op_type)
        if node.op_type in {"Conv", "ConvTranspose", "QLinearConv"}:
            for attr in node.attribute:
                if attr.name == "pads" and any(value < 0 for value in attr.ints):
                    negative_conv_pads.append({"output": node.output[0], "pads": list(attr.ints)})
    for item in model.graph.initializer:
        try:
            array = onnx.numpy_helper.to_array(item)
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                nonfinite.append(item.name)
        except Exception:
            pass
    detail.update(
        nonfinite_initializers=nonfinite,
        negative_conv_pads=negative_conv_pads,
        banned_ops=banned,
        function_count=len(model.functions),
        sparse_initializer_count=len(model.graph.sparse_initializer),
    )
    if nonfinite:
        reasons.append("nonfinite_initializer")
    if negative_conv_pads:
        reasons.append("negative_conv_pads")
    if banned:
        reasons.append("banned_ops")
    if model.functions or model.graph.sparse_initializer:
        reasons.append("function_or_sparse")
    reasons = sorted(set(reasons))
    return not reasons, reasons, detail


SCANNER.strict_extra = strict_extra_schema


def make_session(data: bytes, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = onnx.load_model_from_string(data)
    sanitized = SCANNER.scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known_four(task: int, data: bytes):
    output = {}
    for disabled, mode in ((True, "disable_all"), (False, "default")):
        for threads in (1, 4):
            key = f"{mode}_threads{threads}"
            stats = {"right": 0, "wrong": 0, "errors": 0, "first_error": None}
            try:
                session = make_session(data, disabled, threads)
            except Exception as exc:
                stats["session_error"] = f"{type(exc).__name__}: {exc}"
                output[key] = stats
                continue
            for subset in ("train", "test", "arc-gen"):
                for example in SCANNER.scoring.load_examples(task)[subset]:
                    benchmark = SCANNER.scoring.convert_to_numpy(example)
                    if benchmark is None:
                        continue
                    try:
                        raw = session.run(
                            [session.get_outputs()[0].name],
                            {session.get_inputs()[0].name: benchmark["input"]},
                        )[0]
                        key2 = "right" if np.array_equal(raw > 0, benchmark["output"] > 0) else "wrong"
                        stats[key2] += 1
                    except Exception as exc:
                        stats["errors"] += 1
                        if stats["first_error"] is None:
                            stats["first_error"] = f"{type(exc).__name__}: {exc}"
            output[key] = stats
    return output


SCANNER.known_dual = known_four


if __name__ == "__main__":
    (HERE / "candidates").mkdir(parents=True, exist_ok=True)
    (HERE / "evidence").mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--fresh", "500"]
    raise SystemExit(SCANNER.main())
