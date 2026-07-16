#!/usr/bin/env python3
"""Non-mutating known/fresh audit for the four assigned NeuroGolf models."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from lib import scoring  # noqa: E402


HASHES = {54: "264363fd", 77: "36fdfd69", 118: "50846271", 173: "72322fa7"}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def encode(grid: list[list[int]]) -> np.ndarray:
    value = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            value[0, color, row, col] = 1.0
    return value


def raw_session(model: onnx.ModelProto, optimize: bool):
    model = scoring.sanitize_model(copy.deepcopy(model))
    if model is None:
        raise RuntimeError("model sanitation failed")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )


def run_one(session, example: dict) -> tuple[bool, int, float | None]:
    raw = session.run(["output"], {"input": encode(example["input"])})[0]
    expected = encode(example["output"]) > 0
    near = int(np.count_nonzero((raw > 0) & (raw < 0.25)))
    positive = raw[raw > 0]
    minimum = float(positive.min()) if positive.size else None
    return bool(np.array_equal(raw > 0, expected)), near, minimum


def audit(task: int, path: Path, fresh: int, seeds: list[int]) -> dict:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    dynamic = []
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if value.type.HasField("tensor_type"):
            for dim in value.type.tensor_type.shape.dim:
                if not dim.HasField("dim_value") or dim.dim_value <= 0:
                    dynamic.append(value.name)
                    break
    ops = [node.op_type for node in model.graph.node]
    nested = any(
        attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attr in node.attribute
    )
    bad_bias = []
    init = {item.name: item for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type in {"Conv", "ConvTranspose", "QLinearConv"}:
            bias_index = 2 if node.op_type in {"Conv", "ConvTranspose"} else 8
            if len(node.input) > bias_index and node.input[bias_index] in init:
                bias = init[node.input[bias_index]]
                weight_index = 1 if node.op_type in {"Conv", "ConvTranspose"} else 3
                weight = init.get(node.input[weight_index])
                if weight is not None and bias.dims and weight.dims and bias.dims[0] != weight.dims[0]:
                    bad_bias.append(node.output[0])

    sessions = []
    session_errors = []
    for optimize in (False, True):
        try:
            sessions.append(raw_session(model, optimize))
            session_errors.append(None)
        except Exception as exc:  # retain DISABLE_ALL evidence even if optimized ORT rejects
            session_errors.append(repr(exc))
    if not sessions:
        raise RuntimeError(f"all ORT session modes rejected model: {session_errors}")
    known = json.loads((ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text())
    known_examples = known["train"] + known["test"] + known["arc-gen"]
    known_right = 0
    dual_disagreements = 0
    near_total = 0
    minima = []
    for example in known_examples:
        results = [run_one(session, example) for session in sessions]
        known_right += int(results[0][0])
        dual_disagreements += int(len(results) == 2 and results[0] != results[1])
        near_total += results[0][1]
        if results[0][2] is not None:
            minima.append(results[0][2])

    module = importlib.import_module(f"task_{HASHES[task]}")
    fresh_rows = []
    for seed in seeds:
        random.seed(seed + task)
        right = 0
        generation_errors = 0
        disagreements = 0
        for _ in range(fresh):
            try:
                example = module.generate()
            except Exception:  # generator rejection is not a model error
                generation_errors += 1
                continue
            results = [run_one(session, example) for session in sessions]
            right += int(results[0][0])
            disagreements += int(len(results) == 2 and results[0] != results[1])
            near_total += results[0][1]
            if results[0][2] is not None:
                minima.append(results[0][2])
        total = fresh - generation_errors
        fresh_rows.append(
            {
                "seed": seed + task,
                "right": right,
                "total": total,
                "rate": right / total if total else None,
                "generation_errors": generation_errors,
                "dual_disagreements": disagreements,
            }
        )

    with tempfile.TemporaryDirectory(dir=HERE, prefix=f"score_{task:03d}_") as workdir:
        score = scoring.score_and_verify(
            copy.deepcopy(model), task, workdir, label="audit", require_correct=False
        )

    return {
        "task": task,
        "path": str(path),
        "file_size": path.stat().st_size,
        "nodes": len(model.graph.node),
        "op_histogram": {op: ops.count(op) for op in sorted(set(ops))},
        "banned": sorted({op for op in ops if op.upper() in BANNED or "Sequence" in op}),
        "nested_graphs": nested,
        "dynamic_tensors": dynamic,
        "conv_bias_ub": bad_bias,
        "session_errors": session_errors,
        "known_right": known_right,
        "known_total": len(known_examples),
        "known_rate": known_right / len(known_examples),
        "known_dual_disagreements": dual_disagreements,
        "fresh": fresh_rows,
        "margin_near_count": near_total,
        "margin_min_positive": min(minima) if minima else None,
        "score_measurement": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=sorted(HASHES), required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--fresh", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[910000, 920000])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.task, args.onnx, args.fresh, args.seeds)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
