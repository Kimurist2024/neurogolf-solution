#!/usr/bin/env python3
"""onnx2code — decompile a NeuroGolf ONNX into readable pseudo-Python.

Purpose: turn the cryptic golfed incumbents into a single annotated, topological
listing so a human (or agent) can read the *graph* as straight-line code —
op by op, with shapes, dtypes, and small initializer values inlined. This is a
graph-level decompiler (it shows WHAT ops run, not the high-level ARC rule); use
it to audit whether a net is a general rule or an example-fit heuristic, and to
see exactly which tensor carries the cost.

Usage:
    uv run python scripts/golf/onnx2code.py <path-to.onnx> [--values N] [--cost TASK]

    --values N   inline initializer/Constant values whose element count <= N
                 (default 16); larger ones are summarised by shape+dtype.
    --cost TASK  also run scoring.score_and_verify for task number TASK and print
                 memory/params/cost/score (require_correct=False).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

_DT = onnx.TensorProto.DataType


def _dtype_name(elem_type: int) -> str:
    try:
        return _DT.Name(elem_type)
    except Exception:
        return str(elem_type)


def _shape_of(tt: onnx.TypeProto.Tensor) -> list[int | str]:
    dims: list[int | str] = []
    for d in tt.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
    return dims


def _bytes_of(dims: list[int], elem_type: int) -> int | None:
    if not dims or any((not isinstance(d, int)) or d <= 0 for d in dims):
        return None
    try:
        itemsize = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type)).itemsize
    except Exception:
        return None
    return int(np.prod(dims)) * int(itemsize)


def _fmt_value(arr: np.ndarray, max_elems: int) -> str:
    flat = arr.reshape(-1)
    if flat.size <= max_elems:
        vals = np.array2string(arr, separator=",", threshold=max_elems + 1)
        vals = " ".join(vals.split())
        return f"= {vals}"
    return f"<{list(arr.shape)} {arr.dtype} {flat.size} elems>"


def decompile(model_path: str, max_value_elems: int = 16) -> str:
    model = onnx.load(model_path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    g = model.graph

    # type/shape map for every named tensor we can resolve
    tmap: dict[str, onnx.TypeProto.Tensor] = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        if vi.type.HasField("tensor_type"):
            tmap[vi.name] = vi.type.tensor_type

    def ann(name: str) -> str:
        tt = tmap.get(name)
        if tt is None:
            return name
        dims = _shape_of(tt)
        b = _bytes_of(dims, tt.elem_type) if all(isinstance(d, int) for d in dims) else None
        tag = f"{dims} {_dtype_name(tt.elem_type)}"
        if b is not None:
            tag += f" {b}B"
        return f"{name}:{{{tag}}}"

    lines: list[str] = []
    op = model.opset_import[0].version if model.opset_import else "?"
    lines.append(f"# {Path(model_path).name}  ir={model.ir_version} opset={op}")
    lines.append(f"# inputs : {', '.join(ann(i.name) for i in g.input)}")
    lines.append(f"# outputs: {', '.join(ann(o.name) for o in g.output)}")

    # initializers (= params, cost = element count) sorted by cost
    inits = []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        inits.append((arr.size, init.name, arr))
    inits.sort(reverse=True)
    if inits:
        lines.append("")
        lines.append(f"# --- initializers (params) : {len(inits)} tensors, "
                     f"{sum(s for s, _, _ in inits)} params ---")
        for size, name, arr in inits:
            lines.append(f"{name}: {list(arr.shape)} {arr.dtype} "
                         f"({size}p) {_fmt_value(arr, max_value_elems)}")

    # nodes in given (topological) order
    lines.append("")
    lines.append(f"# --- graph : {len(g.node)} nodes ---")
    for n in g.node:
        outs = ", ".join(ann(o) for o in n.output)
        ins = ", ".join(n.input)
        attrs = []
        for a in n.attribute:
            if a.type == onnx.AttributeProto.INT:
                attrs.append(f"{a.name}={a.i}")
            elif a.type == onnx.AttributeProto.INTS:
                attrs.append(f"{a.name}={list(a.ints)}")
            elif a.type == onnx.AttributeProto.FLOAT:
                attrs.append(f"{a.name}={a.f}")
            elif a.type == onnx.AttributeProto.STRING:
                attrs.append(f"{a.name}={a.s.decode('utf-8', 'replace')}")
            elif a.type == onnx.AttributeProto.TENSOR:
                arr = numpy_helper.to_array(a.t)
                attrs.append(f"{a.name}={_fmt_value(arr, max_value_elems)}")
            else:
                attrs.append(f"{a.name}=<{onnx.AttributeProto.AttributeType.Name(a.type)}>")
        astr = (" {" + ", ".join(attrs) + "}") if attrs else ""
        lines.append(f"{outs} = {n.op_type}({ins}){astr}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--values", type=int, default=16,
                    help="inline initializer/Constant values with <= this many elements")
    ap.add_argument("--cost", type=int, default=None,
                    help="task number; also print score_and_verify cost")
    args = ap.parse_args()

    print(decompile(str(args.onnx), args.values))

    if args.cost is not None:
        from lib import scoring  # noqa: E402
        m = onnx.load(str(args.onnx))
        res = scoring.score_and_verify(m, args.cost, tempfile.mkdtemp(),
                                       require_correct=False)
        print("\n# --- score_and_verify ---")
        print(f"# {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
