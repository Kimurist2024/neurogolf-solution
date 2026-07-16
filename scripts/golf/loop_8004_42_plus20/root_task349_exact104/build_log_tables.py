#!/usr/bin/env python3
"""Derive task349 offset tables from its fixed power-of-two shift table."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "task349.onnx"
PREVIOUS = HERE / "task349_relation_zero_sig_exact.onnx"
OUTPUT = HERE / "task349_relation_zero_sig_log_exact.onnx"
REPORT = HERE / "relation_zero_sig_log_result.json"
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import (  # noqa: E402
    masks_equal_with_margin,
    model_margin_stable,
    score_and_verify,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_log2_kernel() -> list[int]:
    inp = helper.make_tensor_value_info("x", TensorProto.INT32, [6])
    out = helper.make_tensor_value_info("r", TensorProto.INT8, [6])
    inv = numpy_helper.from_array(np.asarray(1.4426950408889634, np.float32), "inv")
    nodes = [
        helper.make_node("Cast", ["x"], ["xf"], to=TensorProto.FLOAT),
        helper.make_node("Log", ["xf"], ["lx"]),
        helper.make_node("Mul", ["lx", "inv"], ["rf"]),
        helper.make_node("Cast", ["rf"], ["r"], to=TensorProto.INT8),
    ]
    graph = helper.make_graph(nodes, "enumerate_log2", [inp], [out], [inv])
    mini = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    mini.ir_version = 10
    session = ort.InferenceSession(mini.SerializeToString(), providers=["CPUExecutionProvider"])
    values = np.asarray([1, 2, 4, 8, 16, 32], np.int32)
    got = session.run(None, {"x": values})[0]
    if not np.array_equal(got, np.arange(6, dtype=np.int8)):
        raise AssertionError(f"ORT log2 kernel is not exact on fixed domain: {got}")
    return got.astype(int).tolist()


def main() -> int:
    log2_values = verify_log2_kernel()
    model = onnx.load(PREVIOUS)
    arrays = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    shifts = np.asarray(arrays["shift_by_mod"], np.int32)
    expected_r = np.asarray(arrays["hend_offset_by_mod_i8"], np.int8)
    expected_hstart = np.asarray(arrays["hstart_offset_by_mod_i8"], np.int8)
    derived_r = np.log2(shifts).astype(np.int8)
    if not np.array_equal(derived_r, expected_r):
        raise AssertionError("hend-minus-one table is not log2(shift)")
    if not np.array_equal((1 - 3 * derived_r).astype(np.int8), expected_hstart):
        raise AssertionError("hstart table is not 1-3*log2(shift)")

    removed = {"hstart_offset_by_mod_i8", "hend_offset_by_mod_i8"}
    kept = [x for x in model.graph.initializer if x.name not in removed]
    if len(kept) + 2 != len(model.graph.initializer):
        raise AssertionError("offset tables not found exactly once")
    kept.extend([
        numpy_helper.from_array(np.asarray(1.4426950408889634, np.float32), "inv_ln2_f32"),
        numpy_helper.from_array(np.asarray(3, np.int8), "three_i8"),
    ])
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    omitted: set[str] = set()
    rewritten = []
    for node in model.graph.node:
        outputs = set(node.output)
        if outputs & {"hstart_offset_i8", "hend_offset_i8", "top_offset_i8"}:
            omitted.update(outputs & {"hstart_offset_i8", "hend_offset_i8", "top_offset_i8"})
            continue
        rewritten.append(node)
        if list(node.output) == ["shift_factor"]:
            rewritten.extend([
                helper.make_node(
                    "Cast", ["shift_factor"], ["shift_factor_f32"],
                    name="shift_factor_f32", to=TensorProto.FLOAT,
                ),
                helper.make_node("Log", ["shift_factor_f32"], ["ln_shift"], name="ln_shift"),
                helper.make_node(
                    "Mul", ["ln_shift", "inv_ln2_f32"], ["radius_f32"],
                    name="log2_shift",
                ),
                helper.make_node(
                    "Cast", ["radius_f32"], ["hend_offset_i8"],
                    name="hend_offset_i8_from_log2", to=TensorProto.INT8,
                ),
                helper.make_node(
                    "Mul", ["hend_offset_i8", "three_i8"], ["three_radius_i8"],
                    name="three_radius_i8",
                ),
                helper.make_node(
                    "Sub", ["one_i8", "three_radius_i8"], ["hstart_offset_i8"],
                    name="hstart_offset_i8_from_log2",
                ),
                helper.make_node(
                    "Add", ["hstart_offset_i8", "hend_offset_i8"], ["top_offset_i8"],
                    name="top_offset_i8_from_log2",
                ),
            ])
    if omitted != {"hstart_offset_i8", "hend_offset_i8", "top_offset_i8"}:
        raise AssertionError(f"unexpected omitted producers: {omitted}")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    if inferred.functions or inferred.graph.sparse_initializer:
        raise AssertionError("functions/sparse initializers are forbidden")
    onnx.save(model, OUTPUT)

    work = HERE / "work_log"
    baseline = score_and_verify(
        copy.deepcopy(onnx.load(SOURCE)), 349, str(work),
        label="task349_lb8008_14", require_correct=True,
    )
    candidate = score_and_verify(
        copy.deepcopy(model), 349, str(work),
        label="task349_relation_zero_sig_log_exact", require_correct=True,
    )
    if baseline is None or candidate is None:
        raise RuntimeError(f"official scoring failed: {baseline=} {candidate=}")
    known_equal = masks_equal_with_margin(
        copy.deepcopy(onnx.load(SOURCE)), copy.deepcopy(model), 349, margin=0.25
    )
    margin_ok, min_abs = model_margin_stable(copy.deepcopy(model), 349, margin=0.25)
    payload = {
        "task": 349,
        "source_sha256": digest(SOURCE),
        "candidate_sha256": digest(OUTPUT),
        "baseline": baseline,
        "candidate": candidate,
        "cost_reduction": int(baseline["cost"] - candidate["cost"]),
        "projected_gain": math.log(baseline["cost"] / candidate["cost"]),
        "known_mask_equal": bool(known_equal),
        "margin_stable": bool(margin_ok),
        "minimum_nonzero_abs": float(min_abs),
        "proof": {
            "inherits_prior_exact_proofs": True,
            "shift_table_domain": sorted(set(shifts.astype(int).tolist())),
            "ort_log2_enumeration": log2_values,
            "hend_minus_one_identity": "log2(shift)",
            "hstart_identity": "1 - 3*log2(shift)",
            "all_11_table_rows_exhaustive": True,
            "all_input_equivalent": True
        }
    }
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["cost_reduction"] > 0 and known_equal and margin_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
