#!/usr/bin/env python3
"""Build numerically alternative task132 gauge-reuse candidates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "submission_base_8000.46.zip"
OUTPUT_DIR = Path(__file__).resolve().parent / "lane_task132_scale"


def attribute(node: onnx.NodeProto, name: str) -> onnx.AttributeProto:
    return next(item for item in node.attribute if item.name == name)


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    initializer = next(item for item in model.graph.initializer if item.name == name)
    initializer.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(value, dtype=np.float32), name=name))


def build(source: onnx.ModelProto, exponent: int, balance: str) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    arrays = {
        item.name: numpy_helper.to_array(item).astype(np.float64)
        for item in model.graph.initializer
    }
    f_gauge = np.array([[0.5, 0.0], [-1.0, 1.0]], dtype=np.float64)
    c_gauge = np.array([[1.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    q_gauged = np.einsum("if,jc,fcqv->ijqv", f_gauge, c_gauge, arrays["Q"])
    pc_gauged = np.einsum("if,fpu->ipu", np.linalg.inv(f_gauge).T, arrays["PC"])
    l_gauged = np.einsum("jc,csw->jsw", np.linalg.inv(c_gauge).T, arrays["L"])

    scale = float(2 ** (exponent // 2))
    q_scale = 10_000_000_000.0 / float(2**exponent)
    q_new = q_gauged * q_scale
    if balance == "pc":
        pc_new, l_new = pc_gauged / q_scale, l_gauged
    elif balance == "l":
        pc_new, l_new = pc_gauged, l_gauged / q_scale
    elif balance == "split":
        root = math.sqrt(q_scale)
        pc_new, l_new = pc_gauged / root, l_gauged / root
    else:
        raise ValueError(balance)
    h_new = arrays["H"] * scale

    repeated = np.einsum("mtmt->mt", q_new).astype(np.float32)
    desired = (arrays["A"] / (scale * scale)).astype(np.float32)
    if not np.array_equal(repeated, desired):
        raise RuntimeError(f"repeated view mismatch for exponent={exponent} balance={balance}")

    replace_initializer(model, "PC", pc_new)
    replace_initializer(model, "Q", q_new)
    replace_initializer(model, "L", l_new)
    replace_initializer(model, "H", h_new)
    kept = [item for item in model.graph.initializer if item.name != "A"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    node = model.graph.node[0]
    lhs, rhs = attribute(node, "equation").s.decode().split("->")
    terms = lhs.split(",")
    names = list(node.input)
    for index, name in enumerate(names):
        if name != "A":
            continue
        names[index] = "Q"
        terms[index] = {"mt": "mtmt", "lR": "lRlR"}[terms[index]]
    del node.input[:]
    node.input.extend(names)
    attribute(node, "equation").s = (",".join(terms) + "->" + rhs).encode()
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASELINE) as archive:
        source = onnx.load_model_from_string(archive.read("task132.onnx"))
    rows = []
    for exponent in range(24, 43, 2):
        for balance in ("pc", "l", "split"):
            try:
                model = build(source, exponent, balance)
                path = OUTPUT_DIR / f"task132_e{exponent}_{balance}.onnx"
                onnx.save(model, path)
                rows.append(
                    {
                        "exponent": exponent,
                        "balance": balance,
                        "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            except Exception as exc:
                rows.append({"exponent": exponent, "balance": balance, "error": repr(exc)})
    manifest = OUTPUT_DIR / "build_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"built": sum("path" in row for row in rows), "errors": sum("error" in row for row in rows), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
