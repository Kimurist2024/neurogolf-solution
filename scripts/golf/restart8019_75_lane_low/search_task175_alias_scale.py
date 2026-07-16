#!/usr/bin/env python3
"""Recover the exact cost-122 task175 selector alias and repair its margin.

The cost-134 source has ``Msel = S @ C`` where C swaps columns zero and one.
After replacing every S occurrence by Msel, the original contraction is
restored by applying that same involution to Q axis 0 and R axes 0 and 2.
The historical alias sweep rejected this exact gauge before checking signs
because float32 Einsum evaluation placed a true logit just below 0.25.
TA is a one-use homogeneous input of the terminal Einsum, so multiplying it
by a positive power of two uniformly scales the output without changing the
classifier.  This script fixes the construction and runs a complete known
screen before emitting it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "scripts/golf/restart8018_91_lane_low/candidates/task175_gauge_remove_w_v.onnx"
OUTPUT = HERE / "candidates/task175_alias122_scale2.onnx"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def swap01(array: np.ndarray, axis: int) -> np.ndarray:
    order = np.arange(array.shape[axis])
    order[0], order[1] = order[1], order[0]
    return np.take(array, order, axis=axis).copy()


def build(scale: float) -> onnx.ModelProto:
    model = onnx.load(SOURCE)
    node = model.graph.node[0]
    if node.op_type != "Einsum" or list(node.output) != ["output"]:
        raise RuntimeError("source terminal Einsum drift")
    replaced = 0
    for index, name in enumerate(node.input):
        if name == "S":
            node.input[index] = "Msel"
            replaced += 1
    if replaced != 7:
        raise RuntimeError(f"expected seven S uses, found {replaced}")

    arrays = {item.name: np.asarray(numpy_helper.to_array(item)).copy()
              for item in model.graph.initializer if item.name != "S"}
    # Msel = S @ C.  C is an involution, so C on the connected axes cancels.
    arrays["Q"] = swap01(arrays["Q"], 0)
    arrays["R"] = swap01(swap01(arrays["R"], 0), 2)
    arrays["TA"] = arrays["TA"] * np.float32(scale)
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError("nonfinite initializer after gauge")
    del model.graph.initializer[:]
    model.graph.initializer.extend(
        numpy_helper.from_array(array, name) for name, array in arrays.items()
    )
    model.producer_name = f"codex-task175-alias122-scale{scale:g}"
    return model


def session(model: onnx.ModelProto, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def known_audit(model: onnx.ModelProto) -> dict[str, object]:
    result: dict[str, object] = {"total": 0, "wrong": 0, "errors": 0,
                                "nonfinite": 0, "small_positive": 0,
                                "min_positive": None, "max_false": None}
    positives: list[float] = []
    false_values: list[float] = []
    sessions = {threads: session(model, threads) for threads in (1, 4)}
    per_threads: dict[str, dict[str, int]] = {}
    examples = scoring.load_examples(175)
    cases = []
    for subset in ("train", "test", "arc-gen"):
        for example in examples.get(subset, []):
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is not None:
                cases.append(benchmark)
    for threads, sess in sessions.items():
        row = {"right": 0, "wrong": 0, "errors": 0}
        for benchmark in cases:
            result["total"] = int(result["total"]) + 1
            try:
                raw = sess.run(["output"], {"input": benchmark["input"]})[0]
            except Exception:
                result["errors"] = int(result["errors"]) + 1
                row["errors"] += 1
                continue
            target = benchmark["output"] > 0.0
            finite = np.isfinite(raw)
            result["nonfinite"] = int(result["nonfinite"]) + int((~finite).sum())
            result["small_positive"] = int(result["small_positive"]) + int(
                ((raw > 0.0) & (raw < 0.25)).sum()
            )
            if target.any():
                positives.append(float(raw[target].min()))
            if (~target).any():
                false_values.append(float(raw[~target].max()))
            if np.array_equal(raw > 0.0, target):
                row["right"] += 1
            else:
                row["wrong"] += 1
                result["wrong"] = int(result["wrong"]) + 1
        per_threads[str(threads)] = row
    result["per_threads"] = per_threads
    result["min_positive"] = min(positives) if positives else None
    result["max_false"] = max(false_values) if false_values else None
    return result


def main() -> int:
    rows = []
    winner = None
    for scale in (1.0, 2.0, 4.0):
        model = build(scale)
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        shape = [int(dim.dim_value) for dim in
                 inferred.graph.output[0].type.tensor_type.shape.dim]
        audit = known_audit(model)
        row = {"scale": scale, "shape": shape, "audit": audit}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if (winner is None and shape == [1, 10, 30, 30]
                and audit["wrong"] == 0 and audit["errors"] == 0
                and audit["nonfinite"] == 0 and audit["small_positive"] == 0
                and float(audit["min_positive"]) >= 0.25
                and float(audit["max_false"]) <= 0.0):
            winner = (scale, model, audit)
    if winner is None:
        raise RuntimeError("no strict known-data winner")

    scale, model, audit = winner
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    blob = model.SerializeToString()
    OUTPUT.write_bytes(blob)
    memory, params, cost = cost_of(str(OUTPUT))
    evidence = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE.read_bytes()),
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "candidate_sha256": sha256(blob),
        "scale": scale,
        "profile": {"memory": memory, "params": params, "cost": cost},
        "known": audit,
        "search": rows,
        "proof": {
            "alias": "Msel = S @ C, C swaps columns 0/1, C @ C = I",
            "compensation": "Q axis0 and R axes0/2 are permuted by C",
            "uniform_scale": "TA occurs once in the sole homogeneous output Einsum",
        },
    }
    (HERE / "task175_alias122_known_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
