#!/usr/bin/env python3
"""Build exact/no-regression candidates for lane 140."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_COST = {5: 2325, 297: 371, 308: 433}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_initializer(model: onnx.ModelProto, name: str) -> None:
    keep = [item for item in model.graph.initializer if item.name != name]
    if len(keep) == len(model.graph.initializer):
        raise RuntimeError(f"missing initializer {name}")
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)


def task297_shared_c_scale() -> tuple[onnx.ModelProto, dict[str, object]]:
    model = onnx.load(HERE / "baseline/task297.onnx")
    inits = {item.name: item for item in model.graph.initializer}
    q_scale = float(numpy_helper.to_array(inits["q_scale"]))
    c_scale = float(numpy_helper.to_array(inits["c_scale"]))
    coeff = numpy_helper.to_array(inits["hash_coeff"])
    if not np.array_equal(coeff, np.asarray([5, 9], dtype=np.uint8).reshape(1, 2, 1, 1)):
        raise RuntimeError("unexpected task297 hash coefficients")
    inits["hash_coeff"].CopyFrom(
        numpy_helper.from_array(np.asarray([34, 61], dtype=np.uint8).reshape(1, 2, 1, 1), "hash_coeff")
    )
    qmm = next(node for node in model.graph.node if node.op_type == "QLinearMatMul")
    for index in (1, 4, 6):
        if qmm.input[index] != "q_scale":
            raise RuntimeError("unexpected task297 QLinearMatMul scale wiring")
        qmm.input[index] = "c_scale"
    remove_initializer(model, "q_scale")

    reachable = np.asarray([0, 1, 2, 4, 10, 18, 24, 26, 27, 28], dtype=np.float64)
    old_products = (reachable[:, None].astype(np.uint16) * np.asarray([5, 9], dtype=np.uint16)[None, :]).astype(np.uint8)
    new_products = (reachable[:, None].astype(np.uint16) * np.asarray([34, 61], dtype=np.uint16)[None, :]).astype(np.uint8)
    old_h = np.rint(old_products.astype(np.float64) * q_scale).astype(np.int64)
    new_h = np.rint(new_products.astype(np.float64) * c_scale).astype(np.int64)
    return model, {
        "removed_initializer": "q_scale",
        "old_q_scale": q_scale,
        "shared_c_scale": c_scale,
        "hash_coeff_before": [5, 9],
        "hash_coeff_after": [34, 61],
        "reachable_codes": reachable.astype(int).tolist(),
        "old_uint8_products": old_products.tolist(),
        "new_uint8_products": new_products.tolist(),
        "old_quantized_features": old_h.tolist(),
        "new_quantized_features": new_h.tolist(),
        "hypothesis_equivalent_over_reachable_codes": bool(np.array_equal(old_h, new_h)),
        "rejection_proof": "The real-valued rescale identity is invalid in the ONNX graph because Mul is uint8: 34*v and 61*v wrap above 255. Quantized features diverge and known gold rejects the candidate.",
    }


def task308_bypass_constant_shape_copy() -> tuple[onnx.ModelProto, dict[str, object]]:
    """Diagnostic exact no-op: bypass the constant vector's identity crop."""
    model = onnx.load(HERE / "baseline/task308.onnx")
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "out_shape4":
                node.input[index] = "out_shape4_const"
    remove_outputs = {"out_shape4_len", "out_shape4"}
    keep_nodes = [node for node in model.graph.node if not remove_outputs.intersection(node.output)]
    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)
    keep_vi = [value for value in model.graph.value_info if value.name not in remove_outputs]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep_vi)
    return model, {
        "proof": "CenterCropPad(out_shape4_const, Shape(out_shape4_const)) is the identity on the fixed [4]-element vector.",
        "removed_nodes": ["Shape:out_shape4_len", "CenterCropPad:out_shape4"],
        "warning": "diagnostic only until runtime/shape-cloak audit passes",
    }


def main() -> None:
    candidates = HERE / "candidates"
    candidates.mkdir(exist_ok=True)
    auditor = load_module(
        "lane140_auditor",
        ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
    )
    harvest = load_module(
        "lane140_harvest",
        ROOT / "scripts/golf/loop_7999_13/lane_harvest/harvest.py",
    )
    builders = {
        "task297_shared_c_scale": (297, task297_shared_c_scale),
        "task308_bypass_constant_shape_copy": (308, task308_bypass_constant_shape_copy),
    }
    output: dict[str, object] = {"base_cost": BASE_COST, "candidates": {}}
    for label, (task, builder) in builders.items():
        model, proof = builder()
        path = candidates / f"{label}.onnx"
        onnx.save(model, path)
        data = path.read_bytes()
        actual = harvest.actual_screen(data, task)
        row = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "proof": proof,
            "actual_cost": actual,
            "strictly_lower": actual is not None and actual < BASE_COST[task],
            "audit": auditor.audit(label, task, path),
        }
        output["candidates"][label] = row
        (HERE / "candidate_audit.json").write_text(json.dumps(output, indent=2) + "\n")
        print(label, actual, row["sha256"], flush=True)


if __name__ == "__main__":
    main()
