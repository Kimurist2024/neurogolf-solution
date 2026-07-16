#!/usr/bin/env python3
"""Rebase-compatible audit for the eight low-cost extension targets."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference

import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TARGETS = (33, 282, 84, 362, 381, 1, 352, 283)
BASE_COST = {33: 96, 282: 96, 84: 92, 362: 92, 381: 92, 1: 90, 352: 90, 283: 89}
PRIOR = ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [int(d.dim_value) if d.HasField("dim_value") else None for d in value.type.tensor_type.shape.dim]


def run_known(model: onnx.ModelProto, task: int, disable: bool) -> dict[str, int]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL if disable else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    examples = scoring.load_examples(task)
    expected_total = sum(len(examples[name]) for name in ("train", "test", "arc-gen"))
    try:
        session = ort.InferenceSession(sanitized.SerializeToString(), options)
    except Exception as exc:
        return {
            "right": 0,
            "wrong": 0,
            "errors": expected_total,
            "skipped": 0,
            "total_executable": expected_total,
            "runtime_output_shapes": [],
            "session_error": f"{type(exc).__name__}: {exc}",
        }
    right = wrong = errors = skipped = 0
    runtime_output_shapes: set[tuple[int, ...]] = set()
    for subset in ("train", "test", "arc-gen"):
        for example in examples[subset]:
            bench = scoring.convert_to_numpy(example)
            if bench is None:
                skipped += 1
                continue
            try:
                raw = session.run(["output"], {"input": bench["input"]})[0]
                runtime_output_shapes.add(tuple(int(x) for x in raw.shape))
                if np.array_equal(raw > 0, bench["output"] > 0):
                    right += 1
                else:
                    wrong += 1
            except Exception:
                errors += 1
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "skipped": skipped,
        "total_executable": right + wrong + errors,
        "runtime_output_shapes": [list(x) for x in sorted(runtime_output_shapes)],
    }


def base_row(task: int) -> dict:
    path = HERE / "base" / f"task{task:03d}.onnx"
    model = onnx.load(path)
    checker = strict = True
    checker_error = strict_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        checker = False
        checker_error = repr(exc)
    try:
        shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:
        strict = False
        strict_error = repr(exc)
    max_einsum = max((len(n.input) for n in model.graph.node if n.op_type == "Einsum"), default=0)
    duplicate_initializers = []
    for i, a in enumerate(model.graph.initializer):
        for b in model.graph.initializer[i + 1 :]:
            if a.data_type == b.data_type and list(a.dims) == list(b.dims) and a.raw_data == b.raw_data:
                duplicate_initializers.append([a.name, b.name])
    disabled = run_known(model, task, True)
    default = run_known(model, task, False)
    declared_output = dims(model.graph.output[0])
    observed = disabled["runtime_output_shapes"]
    output_truthful = len(observed) == 1 and observed[0] == declared_output
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "cost": BASE_COST[task],
        "params": int(scoring.calculate_params(model)),
        "nodes": len(model.graph.node),
        "ops": dict(Counter(n.op_type for n in model.graph.node)),
        "max_einsum_inputs": max_einsum,
        "giant_einsum": max_einsum >= 15,
        "checker_full": checker,
        "checker_error": checker_error,
        "strict_data_prop": strict,
        "strict_error": strict_error,
        "duplicate_initializers": duplicate_initializers,
        "declared_output_shape": declared_output,
        "output_shape_truthful": output_truthful,
        "known_disable_all": disabled,
        "known_default": default,
        "conv_bias_ub": 0,
    }


def main() -> None:
    prior = json.loads(PRIOR.read_text())
    rows = prior["rows"]
    history = {}
    for task in TARGETS:
        selected = [r for r in rows if r.get("task") == task]
        history[str(task)] = {
            "exact_same_baseline_lineage": True,
            "unique_different": len(selected),
            "stages": dict(Counter(r.get("stage") for r in selected)),
            "reasons": dict(Counter(r.get("reason") for r in selected)),
        }
    with zipfile.ZipFile(ROOT / "submission_base_7999.13.zip") as old, zipfile.ZipFile(
        ROOT / "submission_base_8005.16.zip"
    ) as new:
        same = {
            str(task): hashlib.sha256(old.read(f"task{task:03d}.onnx")).hexdigest()
            == hashlib.sha256(new.read(f"task{task:03d}.onnx")).hexdigest()
            for task in TARGETS
        }
    exact = json.loads((HERE / "exact_shave_audit.json").read_text())
    result = {
        "baseline_zip": "submission_base_8005.16.zip",
        "baseline_zip_sha256": hashlib.sha256((ROOT / "submission_base_8005.16.zip").read_bytes()).hexdigest(),
        "targets": list(TARGETS),
        "same_payload_as_exhaustive_7999_13_scan": same,
        "baseline": {str(task): base_row(task) for task in TARGETS},
        "historical_rescreen": history,
        "exact_shaves": exact,
        "accepted": [],
        "aggregate_gain": 0.0,
    }
    (HERE / "audit_results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
