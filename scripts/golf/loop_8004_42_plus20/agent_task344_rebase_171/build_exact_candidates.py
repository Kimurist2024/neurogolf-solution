#!/usr/bin/env python3
"""Build strict-lower task344 reparameterizations from the 8009.46 member.

The authority's positional initializer B has shape [2, 30], but its final
twenty columns are exact zeros.  ONNX sparse initializers preserve the logical
dense shape while storing only nonzero values, so this rewrite does not alter
the Einsum equation or any floating-point arithmetic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline/task344.onnx"
CANDIDATES = HERE / "candidates"
EXPECTED_SHA256 = "05bedf3ca834aadfc973c00fc91cafdb4d0ae1aaab374115d924e2e33fb1bf6c"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_to_sparse(tensor: onnx.TensorProto) -> onnx.SparseTensorProto:
    array = numpy_helper.to_array(tensor)
    coordinates = np.argwhere(array != 0).astype(np.int64)
    values = array[tuple(coordinates.T)]
    values_tensor = numpy_helper.from_array(values, tensor.name)
    indices_tensor = numpy_helper.from_array(coordinates, tensor.name + "_indices")
    return onnx.helper.make_sparse_tensor(values_tensor, indices_tensor, list(array.shape))


def main() -> None:
    if sha256(BASELINE) != EXPECTED_SHA256:
        raise RuntimeError("task344 authority member changed")
    model = onnx.load(BASELINE)
    dense = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    if "B" not in dense or dense["B"].shape != (2, 30):
        raise RuntimeError("unexpected B initializer")
    if np.count_nonzero(dense["B"][:, 10:]) != 0 or np.count_nonzero(dense["B"][:, :10]) != 19:
        raise RuntimeError("B finite support changed")

    rewritten = onnx.ModelProto()
    rewritten.CopyFrom(model)
    kept = [item for item in rewritten.graph.initializer if item.name != "B"]
    del rewritten.graph.initializer[:]
    rewritten.graph.initializer.extend(kept)
    rewritten.graph.sparse_initializer.append(
        dense_to_sparse(next(item for item in model.graph.initializer if item.name == "B"))
    )
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    sparse_output = CANDIDATES / "task344_exact_sparse_B_rejected.onnx"
    onnx.save(rewritten, sparse_output)
    sparse_error = None
    try:
        onnx.checker.check_model(rewritten, full_check=True)
        onnx.shape_inference.infer_shapes(rewritten, strict_mode=True, data_prop=True)
    except Exception as error:  # expected: sparse input rank is not inferred for Einsum
        sparse_error = f"{type(error).__name__}: {error}"

    # Shared-V reparameterization target.  In column-vector notation the
    # authority's pre-center local transform is
    #
    #   G V z,  where G = H^T S^T H.
    #
    # Reusing V as a color Gram basis gives R V z with R = V V^T.  Choose P
    # such that P^T R = G.  The center gating, M, and output V are unchanged,
    # so the real-valued target contraction is algebraically identical.  The
    # serialized float32 P has a recorded nonzero residual and is not claimed
    # to be exact for every float input.
    h, v, s, b, m = (dense[name] for name in ("H", "V", "S", "B", "M"))
    gram = v.astype(np.float64) @ v.astype(np.float64).T
    local = h.astype(np.float64).T @ s.astype(np.float64).T @ h.astype(np.float64)
    p = np.linalg.solve(gram.T, local.T).astype(np.float32)
    residual = p.astype(np.float64).T @ gram - local

    terms = ["ls", "ld", "...dpq"]
    inputs = ["V", "V", "input"]
    for symbol in "abefgijmnrtuvzAB":
        terms += [f"{symbol}p", f"{symbol}h"]
        inputs += ["B", "B"]
    for symbol in "CDEFGHIJKLMNOPQR":
        terms += [f"{symbol}q", f"{symbol}w"]
        inputs += ["B", "B"]
    terms += ["...chw", "kc", "xs", "xk", "ky", "yo"]
    inputs += ["input", "V", "V", "P", "M", "V"]
    equation = ",".join(terms) + "->...ohw"
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Einsum", inputs, ["output"], equation=equation)],
        "task344_exact_shared_v_gram",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 30, 30])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10, 30, 30])],
        [
            numpy_helper.from_array(v, "V"),
            numpy_helper.from_array(b, "B"),
            numpy_helper.from_array(p, "P"),
            numpy_helper.from_array(m, "M"),
        ],
    )
    shared = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 12)], ir_version=10)
    shared.producer_name = "codex-task344-rebase171-exact-shared-v"
    onnx.checker.check_model(shared, full_check=True)
    onnx.shape_inference.infer_shapes(shared, strict_mode=True, data_prop=True)
    shared_output = CANDIDATES / "task344_exact_shared_v_cost132.onnx"
    onnx.save(shared, shared_output)

    # A numerically better near-equivalent absorption removes H/S directly. In
    # row-vector notation the authority maps u=(V z) as
    #
    #   u @ H.T @ S @ H.
    #
    # Store the resulting 4x4 matrix G once.  This has the same 132-element
    # cost as the shared-V form, fewer Einsum operands, and a smaller float32
    # serialization residual.
    g64 = h.astype(np.float64).T @ s.astype(np.float64) @ h.astype(np.float64)
    g = g64.astype(np.float32)
    g_residual = g.astype(np.float64) - g64
    terms = ["ld", "...dpq"]
    inputs = ["V", "input"]
    for symbol in "abefgijmnrtuvzAB":
        terms += [f"{symbol}p", f"{symbol}h"]
        inputs += ["B", "B"]
    for symbol in "CDEFGHIJKLMNOPQR":
        terms += [f"{symbol}q", f"{symbol}w"]
        inputs += ["B", "B"]
    terms += ["...chw", "kc", "lk", "ky", "yo"]
    inputs += ["input", "V", "G", "M", "V"]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Einsum", inputs, ["output"], equation=",".join(terms) + "->...ohw")],
        "task344_compact_g",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 30, 30])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10, 30, 30])],
        [
            numpy_helper.from_array(v, "V"),
            numpy_helper.from_array(b, "B"),
            numpy_helper.from_array(g, "G"),
            numpy_helper.from_array(m, "M"),
        ],
    )
    compact = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 12)], ir_version=10)
    compact.producer_name = "codex-task344-rebase171-compact-g"
    onnx.checker.check_model(compact, full_check=True)
    onnx.shape_inference.infer_shapes(compact, strict_mode=True, data_prop=True)
    compact_output = CANDIDATES / "task344_compact_g_cost132.onnx"
    onnx.save(compact, compact_output)

    manifest = {
        "authority_sha256": EXPECTED_SHA256,
        "shared_v_candidate": str(shared_output.relative_to(HERE.parents[3])),
        "shared_v_candidate_sha256": sha256(shared_output),
        "shared_v_parameter_elements": 132,
        "identity": "P.T @ (V @ V.T) == H.T @ S.T @ H",
        "identity_max_abs_residual_float64_from_serialized_P": float(np.max(np.abs(residual))),
        "compact_g_candidate": str(compact_output.relative_to(HERE.parents[3])),
        "compact_g_candidate_sha256": sha256(compact_output),
        "compact_g_identity_target": "G_target = H.T @ S @ H; serialized float32 G has the residual below",
        "compact_g_max_abs_residual_float64_from_serialized_G": float(np.max(np.abs(g_residual))),
        "dense_B_shape": list(dense["B"].shape),
        "dense_B_elements": int(dense["B"].size),
        "dense_B_nonzero": int(np.count_nonzero(dense["B"])),
        "exact_zero_tail_columns": 20,
        "sparse_candidate": str(sparse_output.relative_to(HERE.parents[3])),
        "sparse_full_checker_error": sparse_error,
        "sparse_decision": "REJECT" if sparse_error else "NEEDS_COST_AUDIT",
    }
    (HERE / "audit/exact_build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
