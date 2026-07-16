#!/usr/bin/env python3
"""Compile the hard-gap rank-8 task192 kernel into a cost-442 ONNX graph."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "candidates/task192_rank8_gap_exact_argmax.onnx"
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name)


def main() -> None:
    factor = np.load(HERE / "rank8_noise_gap_factor.npy").astype(np.float32)
    if factor.shape != (30, 8):
        raise RuntimeError(f"unexpected factor shape {factor.shape}")
    training = json.loads((HERE / "rank8_noise_gap_training.json").read_text())
    negative_max = float(training["best"]["negative_max"])
    positive_min = float(training["best"]["positive_min"])
    threshold = np.float32((negative_max + positive_min) / 2.0)

    # B0=-I, B1=2N.  Summing these rows in the histogram gives -count for
    # background and +count for every foreground channel, so ArgMax axis 0 is
    # the exact most-frequent-nonzero selector without a separate mask tensor.
    basis = np.empty((2, 10), dtype=np.float32)
    basis[0] = -1.0
    basis[1] = 2.0
    basis[1, 0] = 0.0

    # The terminal scalar paths are -base (r0) and 2*predicate (r1), due to
    # the center basis scales above.  These coefficients produce
    # z = threshold*base - predicate.
    # A common scale preserves every sign while clearing the validator's
    # positive-margin floor on the official corpus.
    path_coeff = 10.0 * np.asarray([-threshold, -0.5], dtype=np.float32)
    # Common output route R=e0-S=I-N-S using products B_i * G_j, G=[I,S].
    route_product = np.asarray([[-1.0, 1.0], [-0.5, 0.0]], dtype=np.float32)

    nodes = [
        helper.make_node(
            "Einsum", ["input", "basis"], ["hist"],
            equation="bchw,rc->c", name="hist",
        ),
        helper.make_node(
            "ArgMax", ["hist"], ["selected_i64"], axis=0, keepdims=1,
            name="selected_i64",
        ),
        helper.make_node(
            "OneHot", ["selected_i64", "depth", "onehot_values"], ["selected"],
            axis=-1, name="selected",
        ),
        helper.make_node(
            "Pad", ["selected", "neighbor_pads", "one"], ["neighbor_basis"],
            mode="constant", name="neighbor_basis",
        ),
        helper.make_node(
            "Einsum",
            [
                "input", "basis",
                "input", "neighbor_basis", "factor", "factor",
                "input", "neighbor_basis", "factor", "factor",
                "path_coeff", "route_product", "basis", "neighbor_basis",
            ],
            ["output"],
            equation=(
                "bchw,rc,bdhq,rd,qt,wt,bepw,re,ps,hs,r,ij,io,jo->bohw"
            ),
            name="output",
        ),
    ]
    initializers = [
        tensor("basis", basis),
        tensor("factor", factor),
        tensor("path_coeff", path_coeff),
        tensor("route_product", route_product),
        tensor("neighbor_pads", np.asarray([1, 0, 0, 0], dtype=np.int64)),
        tensor("one", np.asarray(1.0, dtype=np.float32)),
        tensor("depth", np.asarray(10, dtype=np.int64)),
        tensor("onehot_values", np.asarray([0.0, 1.0], dtype=np.float32)),
    ]
    graph = helper.make_graph(
        nodes, "task192_rank8_gap_exact_argmax",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10,
        producer_name="codex-task192-rank8-gap",
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE.write_bytes(model.SerializeToString())
    memory, params, cost = cost_of(str(CANDIDATE))
    result = {
        "path": str(CANDIDATE.relative_to(ROOT)),
        "sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "threshold": float(threshold),
        "training_gap": positive_min - negative_max,
        "profile": {"memory": memory, "params": params, "cost": cost},
    }
    (HERE / "rank8_gap_build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
