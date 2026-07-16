#!/usr/bin/env python3
"""Gauge task132's comparator core so it can also replace color matrix A."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[4]
LANE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave17_candidate_meta.zip"
BASELINE = LANE / "baseline_task132.onnx"
OUTPUT = LANE / "task132_q_reuse_312.onnx"


def attribute(node: onnx.NodeProto, name: str) -> onnx.AttributeProto:
    return next(item for item in node.attribute if item.name == name)


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    initializer = next(item for item in model.graph.initializer if item.name == name)
    initializer.CopyFrom(
        numpy_helper.from_array(np.ascontiguousarray(value, dtype=np.float32), name=name)
    )


def build() -> onnx.ModelProto:
    with zipfile.ZipFile(ARCHIVE) as archive:
        payload = archive.read("task132.onnx")
    BASELINE.write_bytes(payload)
    model = onnx.load_model_from_string(payload)
    node = model.graph.node[0]
    arrays = {
        item.name: numpy_helper.to_array(item).astype(np.float64)
        for item in model.graph.initializer
    }

    # Q' = F Q C^T.  Its repeated-index view Q'[m,t,m,t] is exactly
    # [[0.5, 1], [-1, 0]], i.e. A / 1e10.
    f_gauge = np.array([[0.5, 0.0], [-1.0, 1.0]], dtype=np.float64)
    c_gauge = np.array([[1.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    q_new = np.einsum("if,jc,fcqv->ijqv", f_gauge, c_gauge, arrays["Q"])
    pc_new = np.einsum("if,fpu->ipu", np.linalg.inv(f_gauge).T, arrays["PC"])
    l_new = np.einsum("jc,csw->jsw", np.linalg.inv(c_gauge).T, arrays["L"])
    h_new = arrays["H"] * 100_000.0

    repeated = np.einsum("mtmt->mt", q_new)
    np.testing.assert_array_equal(
        repeated.astype(np.float32),
        (arrays["A"] / 10_000_000_000.0).astype(np.float32),
    )
    original_cmp = np.einsum(
        "fpu,fCqv,Csw->puqvsw", arrays["PC"], arrays["Q"], arrays["L"]
    )
    gauged_cmp = np.einsum("fpu,fCqv,Csw->puqvsw", pc_new, q_new, l_new)
    np.testing.assert_allclose(gauged_cmp, original_cmp, atol=0.0, rtol=0.0)

    replace_initializer(model, "PC", pc_new)
    replace_initializer(model, "Q", q_new)
    replace_initializer(model, "L", l_new)
    replace_initializer(model, "H", h_new)
    kept = [item for item in model.graph.initializer if item.name != "A"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    equation = attribute(node, "equation").s.decode()
    lhs, rhs = equation.split("->")
    terms = lhs.split(",")
    names = list(node.input)
    replacements = 0
    for index, name in enumerate(names):
        if name != "A":
            continue
        names[index] = "Q"
        if terms[index] == "mt":
            terms[index] = "mtmt"
        elif terms[index] == "lR":
            terms[index] = "lRlR"
        else:
            raise AssertionError(terms[index])
        replacements += 1
    assert replacements == 2
    del node.input[:]
    node.input.extend(names)
    attribute(node, "equation").s = (",".join(terms) + "->" + rhs).encode()

    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUTPUT)
    print(OUTPUT)
