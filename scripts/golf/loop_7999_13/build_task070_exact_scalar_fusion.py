#!/usr/bin/env python3
"""Fuse task070's exact global scalar intermediate into the terminal Einsum."""

from __future__ import annotations

import copy
import zipfile
from pathlib import Path

import onnx
from onnx import helper


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "submission_base_7999.13.zip"
OUT = Path(__file__).resolve().parent / "lane_task070_exact" / "task070_exact_scalar_fusion.onnx"


def main() -> None:
    with zipfile.ZipFile(BASE) as archive:
        model = onnx.load_from_string(archive.read("task070.onnx"))

    if len(model.graph.node) != 2:
        raise RuntimeError("unexpected task070 graph")
    first, terminal = model.graph.node
    if first.op_type != "Einsum" or terminal.op_type != "Einsum":
        raise RuntimeError("expected two Einsum nodes")
    if list(first.input) != ["input", "R", "T"] or list(first.output) != ["H"]:
        raise RuntimeError("unexpected scalar producer")

    attrs = {a.name: helper.get_attribute_value(a) for a in first.attribute}
    if attrs.get("equation") != b"bihw,ri,er->e":
        raise RuntimeError("unexpected scalar equation")

    old_eq = helper.get_attribute_value(terminal.attribute[0])
    expected = b"blyw,sl,as,bmyz,tm,dt,bjhx,rj,kr,bihw,qi,qc,e,kq,kade->bchw"
    if old_eq != expected:
        raise RuntimeError(f"unexpected terminal equation: {old_eq!r}")
    old_inputs = list(terminal.input)
    h_index = old_inputs.index("H")
    # H[e] = sum(input[b,n,u,v] * R[o,n] * T[e,o]).  Inline that
    # expression while retaining the shared ``e`` label used by D[k,a,d,e].
    new_inputs = old_inputs[:h_index] + ["input", "R", "T"] + old_inputs[h_index + 1 :]
    new_eq = old_eq.replace(b",e,kq", b",bnuv,on,eo,kq")

    replacement = helper.make_node(
        "Einsum",
        new_inputs,
        list(terminal.output),
        equation=new_eq.decode("ascii"),
        name=terminal.name,
    )
    replacement.doc_string = terminal.doc_string

    candidate = copy.deepcopy(model)
    del candidate.graph.node[:]
    candidate.graph.node.extend([replacement])
    # The producer is gone, so its stale value_info entry must go with it.
    retained_value_info = [item for item in candidate.graph.value_info if item.name != "H"]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(retained_value_info)
    onnx.checker.check_model(candidate, full_check=True)
    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, OUT)
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
