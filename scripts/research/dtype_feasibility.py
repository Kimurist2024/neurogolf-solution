#!/usr/bin/env python
"""Dtype feasibility study (FP16 / BOOL / INT8 / INT32) for ONNX Runtime 1.24 CPU.

NeuroGolf 2026: scorer charges memory = num_elements * dtype_itemsize over
intermediate tensors, so narrower dtypes directly raise score. This script
establishes which (op, dtype) combos actually run on ORT 1.24 CPU with
ORT_DISABLE_ALL at the opsets the 400 snapshot models really use.

Outputs:
  docs/research/dtype-feasibility.md
  docs/research/dtype-feasibility.json
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
import traceback

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

ROOT = "/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
SNAP = os.path.join(ROOT, "artifacts", "research_snapshot")
DOCS = os.path.join(ROOT, "docs", "research")

T = TensorProto
NP_OF = {
    T.FLOAT: np.float32, T.FLOAT16: np.float16, T.BOOL: np.bool_,
    T.INT8: np.int8, T.INT32: np.int32, T.INT64: np.int64,
}
NAME_OF = {
    T.FLOAT: "float32", T.FLOAT16: "float16", T.BOOL: "bool",
    T.INT8: "int8", T.INT32: "int32", T.INT64: "int64",
}
FLOATS = {T.FLOAT, T.FLOAT16}

LOGIC_OPS = {"And", "Or", "Xor", "Not"}
COMPARE_OPS = {"Greater", "Less", "Equal", "GreaterOrEqual", "LessOrEqual"}
ARITH_OPS = {"Add", "Sub", "Mul", "Div", "Max", "Min", "Sum", "Mod",
             "Abs", "Neg", "Sign", "Clip", "CumSum",
             "ReduceMax", "ReduceMin", "ReduceSum", "ArgMax", "ArgMin",
             "MatMul", "TopK"}

DTYPE_PLAN = {}
for _o in LOGIC_OPS:
    DTYPE_PLAN[_o] = [T.BOOL]
for _o in COMPARE_OPS:
    DTYPE_PLAN[_o] = [T.FLOAT16, T.INT8, T.INT32]
DTYPE_PLAN["Equal"] = [T.FLOAT16, T.INT8, T.INT32, T.BOOL]
for _o in ["Add", "Sub", "Mul", "Div", "Max", "Min", "Sum", "Mod", "Abs",
           "Neg", "Sign", "Clip", "CumSum", "ReduceMax", "ReduceMin",
           "ReduceSum", "ArgMax", "ArgMin", "MatMul", "TopK"]:
    DTYPE_PLAN[_o] = [T.FLOAT16, T.INT8, T.INT32]
for _o in ["Relu", "Floor", "Sqrt", "Conv", "ConvTranspose", "AveragePool",
           "GlobalMaxPool", "GridSample", "Resize"]:
    DTYPE_PLAN[_o] = [T.FLOAT16]
DTYPE_PLAN["MaxPool"] = [T.FLOAT16, T.INT8]
DTYPE_PLAN["Resize"] = [T.FLOAT16, T.BOOL, T.INT8]
for _o in ["Reshape", "Squeeze", "Unsqueeze", "Transpose", "Tile", "Concat",
           "Gather", "GatherElements", "GatherND", "Slice", "Pad", "Expand",
           "Flatten", "Identity", "Where", "ScatterND", "ScatterElements",
           "OneHot", "Split", "Constant", "ConstantOfShape"]:
    DTYPE_PLAN[_o] = [T.FLOAT16, T.BOOL, T.INT8]
DTYPE_PLAN["MatMulInteger"] = [T.INT8]


def rand(et, shape, rng):
    if et == T.BOOL:
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    if et in (T.INT8, T.INT32, T.INT64):
        return rng.integers(1, 4, size=shape).astype(NP_OF[et])
    return rng.uniform(0.5, 1.5, size=shape).astype(NP_OF[et])


def vi(name, et, shape=None):
    return helper.make_tensor_value_info(name, et, shape)


def init_of(name, et, values):
    return numpy_helper.from_array(np.asarray(values, dtype=NP_OF[et]), name)


def build_op_test(op, opset, et, rng):
    """Return (model, feeds) for a minimal single-op graph, or None if N/A."""
    nodes, gin, gout, inits, feeds = [], [], [], [], {}

    def inp(name, shape, etype=None):
        e = etype if etype is not None else et
        gin.append(vi(name, e, list(shape)))
        feeds[name] = rand(e, shape, rng)
        return name

    if op in {"Add", "Sub", "Mul", "Div", "Max", "Min", "Sum", "And", "Or", "Xor"}:
        nodes.append(helper.make_node(op, [inp("A", (2, 3)), inp("B", (2, 3))], ["Y"]))
        gout.append(vi("Y", et))
    elif op == "Mod":
        kw = {"fmod": 1} if et in FLOATS else {}
        nodes.append(helper.make_node(op, [inp("A", (2, 3)), inp("B", (2, 3))], ["Y"], **kw))
        gout.append(vi("Y", et))
    elif op in COMPARE_OPS:
        nodes.append(helper.make_node(op, [inp("A", (2, 3)), inp("B", (2, 3))], ["Y"]))
        gout.append(vi("Y", T.BOOL))
    elif op in {"Not", "Abs", "Neg", "Sign", "Relu", "Floor", "Sqrt", "Identity"}:
        nodes.append(helper.make_node(op, [inp("X", (2, 3))], ["Y"]))
        gout.append(vi("Y", et))
    elif op == "Clip":
        if opset >= 11:
            inits += [init_of("mn", et, 0), init_of("mx", et, 2)]
            nodes.append(helper.make_node(op, [inp("X", (2, 3)), "mn", "mx"], ["Y"]))
        else:
            nodes.append(helper.make_node(op, [inp("X", (2, 3))], ["Y"], min=0.0, max=2.0))
        gout.append(vi("Y", et))
    elif op == "CumSum":
        inits.append(numpy_helper.from_array(np.asarray(0, dtype=np.int32), "axis"))
        nodes.append(helper.make_node(op, [inp("X", (4,)), "axis"], ["Y"]))
        gout.append(vi("Y", et))
    elif op in {"ReduceMax", "ReduceMin", "ReduceSum"}:
        axes_as_input = (opset >= 13) if op == "ReduceSum" else (opset >= 18)
        if axes_as_input:
            inits.append(init_of("axes", T.INT64, [0]))
            nodes.append(helper.make_node(op, [inp("X", (2, 3)), "axes"], ["Y"]))
        else:
            nodes.append(helper.make_node(op, [inp("X", (2, 3))], ["Y"], axes=[0]))
        gout.append(vi("Y", et))
    elif op in {"ArgMax", "ArgMin"}:
        nodes.append(helper.make_node(op, [inp("X", (2, 3))], ["Y"], axis=0))
        gout.append(vi("Y", T.INT64))
    elif op == "Reshape":
        inits.append(init_of("shape", T.INT64, [3, 2]))
        nodes.append(helper.make_node(op, [inp("X", (2, 3)), "shape"], ["Y"]))
        gout.append(vi("Y", et))
    elif op in {"Squeeze", "Unsqueeze"}:
        shape = (1, 2) if op == "Squeeze" else (2,)
        if opset >= 13:
            inits.append(init_of("axes", T.INT64, [0]))
            nodes.append(helper.make_node(op, [inp("X", shape), "axes"], ["Y"]))
        else:
            nodes.append(helper.make_node(op, [inp("X", shape)], ["Y"], axes=[0]))
        gout.append(vi("Y", et))
    elif op == "Transpose":
        nodes.append(helper.make_node(op, [inp("X", (2, 3))], ["Y"], perm=[1, 0]))
        gout.append(vi("Y", et))
    elif op == "Tile":
        inits.append(init_of("reps", T.INT64, [1, 2]))
        nodes.append(helper.make_node(op, [inp("X", (2, 2)), "reps"], ["Y"]))
        gout.append(vi("Y", et))
    elif op == "Concat":
        nodes.append(helper.make_node(op, [inp("A", (2, 2)), inp("B", (2, 2))], ["Y"], axis=0))
        gout.append(vi("Y", et))
    elif op == "Gather":
        inits.append(init_of("idx", T.INT64, [0, 2]))
        nodes.append(helper.make_node(op, [inp("X", (3, 2)), "idx"], ["Y"], axis=0))
        gout.append(vi("Y", et))
    elif op == "GatherElements":
        inits.append(init_of("idx", T.INT64, [0, 2]))
        nodes.append(helper.make_node(op, [inp("X", (3,)), "idx"], ["Y"], axis=0))
        gout.append(vi("Y", et))
    elif op == "GatherND":
        inits.append(init_of("idx", T.INT64, [[0], [1]]))
        nodes.append(helper.make_node(op, [inp("X", (2, 2)), "idx"], ["Y"]))
        gout.append(vi("Y", et))
    elif op == "Slice":
        if opset >= 10:
            inits += [init_of("st", T.INT64, [0, 0]), init_of("en", T.INT64, [2, 3]),
                      init_of("ax", T.INT64, [0, 1])]
            nodes.append(helper.make_node(op, [inp("X", (4, 4)), "st", "en", "ax"], ["Y"]))
        else:
            nodes.append(helper.make_node(op, [inp("X", (4, 4))], ["Y"],
                                          starts=[0, 0], ends=[2, 3], axes=[0, 1]))
        gout.append(vi("Y", et))
    elif op == "Pad":
        if opset >= 11:
            inits += [init_of("pads", T.INT64, [0, 1, 0, 1]), init_of("cv", et, 0)]
            nodes.append(helper.make_node(op, [inp("X", (2, 3)), "pads", "cv"], ["Y"],
                                          mode="constant"))
        else:
            nodes.append(helper.make_node(op, [inp("X", (2, 3))], ["Y"],
                                          pads=[0, 1, 0, 1], value=0.0, mode="constant"))
        gout.append(vi("Y", et))
    elif op == "Expand":
        inits.append(init_of("shape", T.INT64, [2, 3]))
        nodes.append(helper.make_node(op, [inp("X", (1, 3)), "shape"], ["Y"]))
        gout.append(vi("Y", et))
    elif op == "Flatten":
        nodes.append(helper.make_node(op, [inp("X", (2, 3))], ["Y"], axis=1))
        gout.append(vi("Y", et))
    elif op == "Where":
        nodes.append(helper.make_node(op, [inp("C", (2, 3), T.BOOL),
                                           inp("X", (2, 3)), inp("Y2", (2, 3))], ["Y"]))
        gout.append(vi("Y", et))
    elif op == "ScatterND":
        inits.append(init_of("idx", T.INT64, [[0], [2]]))
        nodes.append(helper.make_node(op, [inp("D", (4,)), "idx", inp("U", (2,))], ["Y"]))
        gout.append(vi("Y", et))
    elif op == "ScatterElements":
        inits.append(init_of("idx", T.INT64, [0, 2]))
        nodes.append(helper.make_node(op, [inp("D", (3,)), "idx", inp("U", (2,))], ["Y"], axis=0))
        gout.append(vi("Y", et))
    elif op == "OneHot":
        inits += [numpy_helper.from_array(np.asarray(3, dtype=np.int64), "depth"),
                  init_of("vals", et, [0, 1])]
        nodes.append(helper.make_node(op, [inp("idx", (3,), T.INT64), "depth", "vals"], ["Y"]))
        gout.append(vi("Y", et))
    elif op == "TopK":
        if opset >= 10:
            inits.append(init_of("K", T.INT64, [2]))
            nodes.append(helper.make_node(op, [inp("X", (3, 4)), "K"], ["V", "I"]))
        else:
            nodes.append(helper.make_node(op, [inp("X", (3, 4))], ["V", "I"], k=2))
        gout += [vi("V", et), vi("I", T.INT64)]
    elif op == "Split":
        if opset >= 13:
            inits.append(init_of("split", T.INT64, [2, 2]))
            nodes.append(helper.make_node(op, [inp("X", (4,)), "split"], ["Y1", "Y2"], axis=0))
        else:
            nodes.append(helper.make_node(op, [inp("X", (4,))], ["Y1", "Y2"], axis=0, split=[2, 2]))
        gout += [vi("Y1", et), vi("Y2", et)]
    elif op == "Conv":
        inits.append(init_of("W", et, rand(et, (1, 1, 3, 3), rng)))
        nodes.append(helper.make_node(op, [inp("X", (1, 1, 4, 4)), "W"], ["Y"]))
        gout.append(vi("Y", et))
    elif op == "ConvTranspose":
        inits.append(init_of("W", et, rand(et, (1, 1, 3, 3), rng)))
        nodes.append(helper.make_node(op, [inp("X", (1, 1, 3, 3)), "W"], ["Y"]))
        gout.append(vi("Y", et))
    elif op in {"MaxPool", "AveragePool"}:
        nodes.append(helper.make_node(op, [inp("X", (1, 1, 4, 4))], ["Y"], kernel_shape=[2, 2]))
        gout.append(vi("Y", et))
    elif op == "GlobalMaxPool":
        nodes.append(helper.make_node(op, [inp("X", (1, 1, 4, 4))], ["Y"]))
        gout.append(vi("Y", et))
    elif op == "GridSample":
        grid = (rng.uniform(-1, 1, size=(1, 2, 2, 2))).astype(NP_OF[et])
        gin.append(vi("G", et, [1, 2, 2, 2]))
        feeds["G"] = grid
        nodes.append(helper.make_node(op, [inp("X", (1, 1, 4, 4)), "G"], ["Y"], mode="nearest"))
        gout.append(vi("Y", et))
    elif op == "MatMul":
        inits.append(init_of("B", et, rand(et, (3, 2), rng)))
        nodes.append(helper.make_node(op, [inp("A", (2, 3)), "B"], ["Y"]))
        gout.append(vi("Y", et))
    elif op == "MatMulInteger":
        inits.append(init_of("B", T.INT8, rng.integers(0, 3, size=(3, 2))))
        nodes.append(helper.make_node(op, [inp("A", (2, 3), T.INT8), "B"], ["Y"]))
        gout.append(vi("Y", T.INT32))
    elif op == "Constant":
        val = numpy_helper.from_array(rand(et, (2, 2), rng), "cval")
        nodes.append(helper.make_node(op, [], ["Y"], value=val))
        gout.append(vi("Y", et))
        # Constant-only graphs need at least one input for ORT? They don't, but
        # keep an unused input to avoid empty-feed edge cases.
        inp("unused", (1,), T.FLOAT)
        nodes.append(helper.make_node("Identity", ["unused"], ["unused_out"]))
        gout.append(vi("unused_out", T.FLOAT))
    elif op == "ConstantOfShape":
        inits.append(init_of("shape", T.INT64, [2, 2]))
        val = numpy_helper.from_array(np.asarray([1], dtype=NP_OF[et]), "cv")
        nodes.append(helper.make_node(op, ["shape"], ["Y"], value=val))
        gout.append(vi("Y", et))
        inp("unused", (1,), T.FLOAT)
        nodes.append(helper.make_node("Identity", ["unused"], ["unused_out"]))
        gout.append(vi("unused_out", T.FLOAT))
    elif op == "Resize":
        if opset >= 11:
            inits.append(numpy_helper.from_array(
                np.asarray([1, 1, 2, 2], dtype=np.float32), "scales"))
            nodes.append(helper.make_node(op, [inp("X", (1, 1, 4, 4)), "", "scales"], ["Y"],
                                          mode="nearest"))
        else:
            inits.append(numpy_helper.from_array(
                np.asarray([1, 1, 2, 2], dtype=np.float32), "scales"))
            nodes.append(helper.make_node(op, [inp("X", (1, 1, 4, 4)), "scales"], ["Y"],
                                          mode="nearest"))
        gout.append(vi("Y", et))
    else:
        return None

    graph = helper.make_graph(nodes, f"test_{op}", gin, gout, inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8 if opset <= 18 else 9
    return model, feeds


def classify_error(exc):
    s = str(exc)
    if "Could not find an implementation" in s or "NOT_IMPLEMENTED" in s:
        return "KERNEL_NOT_FOUND"
    if "INVALID_GRAPH" in s or "Type Error" in s or "INVALID_ARGUMENT" in s:
        return "TYPE_INVALID"
    return "OTHER_ERROR"


def run_model(model, feeds):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 4
    sess = ort.InferenceSession(model.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def inventory():
    files = sorted(glob.glob(os.path.join(SNAP, "task*.onnx")))
    opset_hist = collections.Counter()
    op_counts = collections.Counter()
    op_opsets = collections.defaultdict(set)
    per_model = {}
    for f in files:
        m = onnx.load(f)
        v = [oi.version for oi in m.opset_import if oi.domain in ("", "ai.onnx")][0]
        opset_hist[v] += 1
        seen = sorted(set(n.op_type for n in m.graph.node))
        per_model[os.path.basename(f)] = {"opset": v, "ops": seen,
                                          "num_nodes": len(m.graph.node)}
        for o in seen:
            op_counts[o] += 1
            op_opsets[o].add(v)
    return files, opset_hist, op_counts, op_opsets, per_model


def test_ops(op_counts, op_opsets):
    rng = np.random.default_rng(0)
    results = {}
    for op in sorted(op_counts):
        if op == "Cast":
            results[op] = {"note": "tested separately (see cast_tests)"}
            continue
        plan = DTYPE_PLAN.get(op)
        if plan is None:
            results[op] = {"UNTESTED": "no builder"}
            continue
        results[op] = {}
        for et in plan:
            per_opset = {}
            for opset in sorted(op_opsets[op]):
                built = build_op_test(op, opset, et, rng)
                if built is None:
                    per_opset[opset] = {"status": "N/A"}
                    continue
                model, feeds = built
                try:
                    run_model(model, feeds)
                    per_opset[opset] = {"status": "OK"}
                except Exception as e:  # noqa: BLE001
                    per_opset[opset] = {"status": classify_error(e),
                                        "error": str(e).splitlines()[0][:220]}
            results[op][NAME_OF[et]] = per_opset
    return results


def test_cast(op_opsets):
    rng = np.random.default_rng(1)
    pairs = [(T.FLOAT, T.FLOAT16), (T.FLOAT16, T.FLOAT), (T.FLOAT, T.BOOL),
             (T.BOOL, T.FLOAT), (T.FLOAT, T.INT8)]
    out = {}
    for src, dst in pairs:
        key = f"{NAME_OF[src]}->{NAME_OF[dst]}"
        out[key] = {}
        for opset in sorted(op_opsets.get("Cast", {9, 13, 17})):
            node = helper.make_node("Cast", ["X"], ["Y"], to=dst)
            g = helper.make_graph([node], "cast_test",
                                  [vi("X", src, [2, 3])], [vi("Y", dst)])
            m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
            m.ir_version = 8 if opset <= 18 else 9
            feeds = {"X": rand(src, (2, 3), rng)}
            try:
                run_model(m, feeds)
                out[key][opset] = {"status": "OK"}
            except Exception as e:  # noqa: BLE001
                out[key][opset] = {"status": classify_error(e),
                                   "error": str(e).splitlines()[0][:220]}
    return out


def convert_model_to_fp16(model):
    """Convert FLOAT initializers/Constants/Cast targets to FLOAT16; keep
    graph IO float32 via boundary Casts. Resize roi/scales stay float32."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph

    keep_fp32 = set()  # tensor names that must stay float32 (Resize roi/scales)
    for node in g.node:
        if node.op_type == "Resize":
            for i, name in enumerate(node.input):
                if i >= 1 and name:
                    keep_fp32.add(name)

    for init in g.initializer:
        if init.data_type == T.FLOAT and init.name not in keep_fp32:
            arr = numpy_helper.to_array(init).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))

    for node in g.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == T.FLOAT:
                arr = numpy_helper.to_array(attr.t).astype(np.float16)
                name = attr.t.name
                attr.t.CopyFrom(numpy_helper.from_array(arr, name))
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == T.FLOAT:
                    attr.i = T.FLOAT16

    del g.value_info[:]  # drop stale float32 type annotations

    in_name = g.input[0].name
    out_name = g.output[0].name
    out_et = g.output[0].type.tensor_type.elem_type
    # Graph input stays float32 (competition IF): cast to fp16 for consumers.
    for node in g.node:
        node.input[:] = ["__in_f16" if x == in_name else x for x in node.input]
    cast_in = helper.make_node("Cast", [in_name], ["__in_f16"],
                               to=T.FLOAT16, name="__cast_in_f16")
    pre = [cast_in]
    post = []
    # Only the FLOAT-declared output changes dtype (fp32->fp16); cast it back.
    # BOOL/UINT8/INT8/FLOAT16 outputs are untouched by the conversion.
    if out_et == T.FLOAT:
        for node in g.node:
            node.output[:] = ["__out_f16" if x == out_name else x
                              for x in node.output]
        post.append(helper.make_node("Cast", ["__out_f16"], [out_name],
                                     to=T.FLOAT, name="__cast_out_f32"))
    new_nodes = pre + list(g.node) + post
    del g.node[:]
    g.node.extend(new_nodes)
    return strip_noop_casts(m)


def strip_noop_casts(m):
    """Remove Cast nodes whose input dtype already equals the target dtype.

    After fp32->fp16 conversion, pre-existing Cast(to=FLOAT16) nodes become
    no-ops; leaving them in trips ORT's mandatory InsertCastTransformer
    (type mismatch on InsertedPrecisionFreeCast_*) even with ORT_DISABLE_ALL.
    """
    g = m.graph
    try:
        inferred = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    except Exception:  # noqa: BLE001
        return m
    tmap = {}
    for v in (list(inferred.graph.value_info) + list(inferred.graph.input)
              + list(inferred.graph.output)):
        if v.type.tensor_type.elem_type:
            tmap[v.name] = v.type.tensor_type.elem_type
    for init in g.initializer:
        tmap[init.name] = init.data_type
    out_names = {o.name for o in g.output}
    rename = {}

    def resolve(name):
        while name in rename:
            name = rename[name]
        return name

    keep = []
    for n in g.node:
        n.input[:] = [resolve(x) for x in n.input]
        if n.op_type == "Cast":
            to = next(a.i for a in n.attribute if a.name == "to")
            src = n.input[0]
            if tmap.get(src) == to:
                if n.output[0] in out_names:
                    keep.append(helper.make_node("Identity", [src],
                                                 [n.output[0]]))
                else:
                    rename[n.output[0]] = src
                continue
        keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    return m


def e2e_one(path):
    rng = np.random.default_rng(42)
    x = rng.integers(0, 2, size=(1, 10, 30, 30)).astype(np.float32)
    name = os.path.basename(path)
    entry = {"model": name}
    try:
        base = onnx.load(path)
        out_et = base.graph.output[0].type.tensor_type.elem_type
        entry["declared_output_dtype"] = NAME_OF.get(out_et, str(out_et))
        entry["num_nodes"] = len(base.graph.node)
        ref = run_model(base, {"input": x})[0]
        fp16 = convert_model_to_fp16(base)
        got = run_model(fp16, {"input": x})[0]
        ref_a = np.asarray(ref).astype(np.float64)
        got_a = np.asarray(got).astype(np.float64)
        entry.update({
            "runs": True,
            "output_shape": list(np.asarray(ref).shape),
            "max_abs_diff": float(np.max(np.abs(ref_a - got_a))) if ref_a.size else 0.0,
            "masks_identical": bool(np.array_equal(ref_a > 0.0, got_a > 0.0)),
        })
    except Exception as e:  # noqa: BLE001
        entry.update({"runs": False, "error": str(e).splitlines()[0][:300]})
    return entry


def aggregate_status(per_opset):
    statuses = {v["status"] for v in per_opset.values()}
    if len(statuses) == 1:
        return next(iter(statuses))
    return "MIXED(" + ",".join(
        f"{k}:{v['status']}" for k, v in sorted(per_opset.items())) + ")"


def main():
    os.makedirs(DOCS, exist_ok=True)
    files, opset_hist, op_counts, op_opsets, per_model = inventory()
    print(f"models: {len(files)}; opset hist: {dict(sorted(opset_hist.items()))}",
          flush=True)

    op_results = test_ops(op_counts, op_opsets)
    cast_results = test_cast(op_opsets)

    # e2e: task005 (bool output) plus one mid-size FLOAT-output model so that
    # the cast-back-to-float32 path is exercised too.
    e2e = [e2e_one(os.path.join(SNAP, "task005.onnx"))]
    sized = sorted(per_model.items(), key=lambda kv: kv[1]["num_nodes"])
    for fname, info in sized:
        if fname == "task005.onnx" or not (50 <= info["num_nodes"] <= 600):
            continue
        mm = onnx.load(os.path.join(SNAP, fname))
        if mm.graph.output[0].type.tensor_type.elem_type == T.FLOAT:
            e2e.append(e2e_one(os.path.join(SNAP, fname)))
            break

    fp16_fail = sorted(op for op, dt in op_results.items()
                       if "float16" in dt and any(
                           v["status"] == "KERNEL_NOT_FOUND"
                           for v in dt["float16"].values()))
    bool_int8_fail = {}
    for op, dt in op_results.items():
        for d in ("bool", "int8", "int32"):
            if d in dt and any(v["status"] == "KERNEL_NOT_FOUND"
                               for v in dt[d].values()):
                bool_int8_fail.setdefault(op, []).append(d)

    payload = {
        "ort_version": ort.__version__,
        "onnx_version": onnx.__version__,
        "num_models": len(files),
        "opset_histogram": {str(k): v for k, v in sorted(opset_hist.items())},
        "op_model_counts": dict(sorted(op_counts.items())),
        "op_opsets": {k: sorted(v) for k, v in sorted(op_opsets.items())},
        "op_dtype_support": op_results,
        "fp16_kernel_not_found": fp16_fail,
        "bool_int8_int32_kernel_not_found": bool_int8_fail,
        "cast_tests": cast_results,
        "e2e_fp16_conversion": e2e,
    }
    with open(os.path.join(DOCS, "dtype-feasibility.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    # ---- markdown report ----
    lines = ["# Dtype Feasibility (FP16 / BOOL / INT8) — ONNX Runtime "
             f"{ort.__version__} CPU, ORT_DISABLE_ALL",
             "",
             f"Snapshot: `artifacts/research_snapshot` ({len(files)} models). "
             f"onnx {onnx.__version__}, numpy {np.__version__}.",
             "",
             "## Opset histogram (domain '')", "",
             "| opset | models |", "|---|---|"]
    for k, v in sorted(opset_hist.items()):
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Op usage and dtype support", "",
              "Status per dtype, aggregated over every opset version that the "
              "op actually appears at. `KERNEL_NOT_FOUND` = ORT raised "
              "NOT_IMPLEMENTED / 'Could not find an implementation' — dtype "
              "unusable for that op. `TYPE_INVALID` = ONNX type system "
              "rejects the dtype (not an ORT gap).", "",
              "| op | models | opsets | " +
              " | ".join(["float16", "bool", "int8", "int32"]) + " |",
              "|---|---|---|---|---|---|---|"]
    for op in sorted(op_counts):
        dt = op_results[op]
        cells = []
        for d in ("float16", "bool", "int8", "int32"):
            cells.append(aggregate_status(dt[d]) if d in dt else "—")
        lines.append(f"| {op} | {op_counts[op]} | "
                     f"{','.join(map(str, sorted(op_opsets[op])))} | "
                     + " | ".join(cells) + " |")
    lines += ["", "### Ops failing FP16 (KERNEL_NOT_FOUND)", ""]
    if fp16_fail:
        for op in fp16_fail:
            det = op_results[op]["float16"]
            bad = [str(k) for k, v in sorted(det.items())
                   if v["status"] == "KERNEL_NOT_FOUND"]
            ok = [str(k) for k, v in sorted(det.items()) if v["status"] == "OK"]
            lines.append(f"- **{op}**: fails at opset {','.join(bad)}"
                         + (f"; OK at {','.join(ok)}" if ok else " (all opsets)"))
    else:
        lines.append("- none")
    lines += ["", "### Ops failing BOOL / INT8 / INT32 (KERNEL_NOT_FOUND)", ""]
    if bool_int8_fail:
        for op, ds in sorted(bool_int8_fail.items()):
            lines.append(f"- **{op}**: {', '.join(ds)}")
    else:
        lines.append("- none")
    lines += ["", "## Cast support", "",
              "| cast | result (per opset) |", "|---|---|"]
    for key, per in cast_results.items():
        agg = aggregate_status(per)
        lines.append(f"| {key} | {agg} |")
    lines += ["", "## End-to-end FP16 conversion sanity test", "",
              "Converter: FLOAT initializers/Constant tensors -> FLOAT16, "
              "Cast(to=FLOAT) -> Cast(to=FLOAT16), boundary Cast after "
              "'input' (float32 stays per competition IF), Cast back to "
              "float32 before 'output' only when the declared output dtype "
              "is FLOAT, Resize roi/scales kept float32, and no-op "
              "Cast(fp16->fp16) nodes stripped (leaving them in trips ORT's "
              "mandatory InsertCastTransformer with a type-mismatch error "
              "even at ORT_DISABLE_ALL).", "",
              "Note: the snapshot is heterogeneous — declared output dtypes "
              "are float16 (142), float32 (141), bool (91), uint8 (25), "
              "int8 (1); many models are already fp16-internal.", "",
              "```json", json.dumps(e2e, indent=2), "```", "",
              "## Caveats", "",
              "- An fp16 'OK' means ORT executes the graph; for ops without "
              "a native fp16 CPU kernel ORT may internally wrap the node "
              "with InsertedPrecisionFreeCast (this transformer runs even "
              "at ORT_DISABLE_ALL). That affects runtime only — the scorer "
              "computes memory from the model proto via shape inference, so "
              "fp16 intermediates count 2 bytes per element regardless.",
              "- TYPE_INVALID entries are ONNX type-system limits at that "
              "opset (e.g., Equal fp16/int8 needs opset>=11, CumSum fp16 "
              "needs opset>=14, Pad bool needs opset>=13, Pad int8 needs "
              "opset>=11, Clip int needs opset>=12, MatMul int8 is never "
              "valid — use MatMulInteger). Bumping a model's opset import "
              "lifts these.", ""]
    with open(os.path.join(DOCS, "dtype-feasibility.md"), "w") as fh:
        fh.write("\n".join(lines))

    print("FP16 FAIL:", fp16_fail)
    print("BOOL/INT FAIL:", bool_int8_fail)
    print("CAST:", {k: aggregate_status(v) for k, v in cast_results.items()})
    print("E2E:", json.dumps(e2e))
    untested = [op for op, dt in op_results.items() if "UNTESTED" in dt]
    if untested:
        print("UNTESTED OPS:", untested)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
