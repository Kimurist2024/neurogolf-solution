"""Static Conv/ConvTranspose/QLinearConv bias-length UB checker (shape-inference).

Per Kaggle discussion #699840 (host-confirmed root cause of scoring
non-determinism): a Conv-family node whose bias length < out_channels passes
check_model, infer_shapes, and runs in ORT with no exception, yet reads
out-of-bounds (uninitialised heap) memory for the missing channels -> the same
bytes score differently depending on what ran before in the process (same zip,
different LB score).

CRITICAL: out_channels must be taken from the *inferred output shape*, not from
the weight initializer dims. Nets that COMPUTE their conv weight at runtime
(dynamic weight, not an initializer) hide the true out_channels from a
weight-only check -- exactly how task073/322/372 slipped through for weeks. Use
the shape-inferred channel dim of the node output.

Repair: zero-extend every short bias to out_channels (1-D [out_ch]). Cost goes
UP (adds params); it does not go down.
"""
import sys, zipfile, os
import numpy as np
import onnx
from onnx import shape_inference, numpy_helper


def _out_ch_map(model):
    try:
        inf = shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        return {}
    out = {}
    for vi in list(inf.graph.value_info) + list(inf.graph.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if len(dims) >= 2 and dims[1] > 0:
            out[vi.name] = dims[1]
    return out


def check_model(model):
    """Return [(op, bias_name, bias_len, out_ch), ...]; empty means clean."""
    init = {t.name: t for t in model.graph.initializer}
    ocm = _out_ch_map(model)
    bad = []
    for n in model.graph.node:
        bidx = 8 if n.op_type == "QLinearConv" else (2 if n.op_type in ("Conv", "ConvTranspose") else None)
        if bidx is None or len(n.input) <= bidx or not n.input[bidx]:
            continue
        bn = n.input[bidx]
        if bn not in init:
            continue  # dynamic bias: cannot check statically
        bl = numpy_helper.to_array(init[bn]).size
        oc = ocm.get(n.output[0])
        if oc is not None and bl != oc:
            bad.append((n.op_type, bn, int(bl), int(oc)))
    return bad


def zero_extend_bias(model):
    """Return (model, n_fixed) with every short Conv-family bias zero-extended
    to a 1-D tensor of length out_channels."""
    init = {t.name: t for t in model.graph.initializer}
    ocm = _out_ch_map(model)
    n_fixed = 0
    for n in model.graph.node:
        bidx = 8 if n.op_type == "QLinearConv" else (2 if n.op_type in ("Conv", "ConvTranspose") else None)
        if bidx is None or len(n.input) <= bidx or not n.input[bidx]:
            continue
        bn = n.input[bidx]
        oc = ocm.get(n.output[0])
        if bn not in init or oc is None:
            continue
        arr = numpy_helper.to_array(init[bn])
        flat = arr.flatten()
        if flat.size < oc:
            flat = np.concatenate([flat, np.zeros((oc - flat.size,), dtype=arr.dtype)])
            init[bn].CopyFrom(numpy_helper.from_array(flat.reshape(oc), bn))
            n_fixed += 1
    return model, n_fixed


if __name__ == "__main__":
    z = zipfile.ZipFile(sys.argv[1])
    total = 0
    for nm in z.namelist():
        if not nm.endswith(".onnx"):
            continue
        bad = check_model(onnx.load_model_from_string(z.read(nm)))
        if bad:
            total += 1
            print(f"{os.path.basename(nm)[:7]}: {bad}")
    print(f"\n{total} task(s) with Conv bias-length UB in {sys.argv[1]}")
