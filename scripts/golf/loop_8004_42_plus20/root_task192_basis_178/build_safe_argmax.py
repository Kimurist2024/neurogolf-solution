#!/usr/bin/env python3
"""Replace the rejected Hardmax carrier with exact ArgMax plus OneHot."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "candidates/task192_shared_basis_hardmax.onnx"
OUTPUT = HERE / "candidates/task192_shared_basis_argmax.onnx"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def main() -> None:
    model = onnx.load(SOURCE)
    nodes = list(model.graph.node)
    assert nodes[1].op_type == "Hardmax" and list(nodes[1].output) == ["selected"]
    argmax = helper.make_node(
        "ArgMax",
        ["hist"],
        ["selected_i64"],
        axis=1,
        keepdims=0,
        name="selected_i64",
    )
    onehot = helper.make_node(
        "OneHot",
        ["selected_i64", "depth", "onehot_values"],
        ["selected"],
        axis=-1,
        name="selected",
    )
    del model.graph.node[:]
    model.graph.node.extend([nodes[0], argmax, onehot, *nodes[2:]])
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray(10, dtype=np.int64), "depth"),
            numpy_helper.from_array(
                np.asarray([0.0, 1.0], dtype=np.float32), "onehot_values"
            ),
        ]
    )
    model.graph.value_info.extend(
        [helper.make_tensor_value_info("selected_i64", TensorProto.INT64, [1])]
    )
    model.producer_name = "codex-task192-basis178-safe-argmax"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUTPUT)
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": digest(SOURCE),
        "source_profile": profile(SOURCE),
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "candidate_sha256": digest(OUTPUT),
        "candidate_profile": profile(OUTPUT),
        "classification": "SAFE_NO_HARDMAX_ARGMAX_ONEHOT_SHARED_BASIS",
        "proof": "ArgMax keepdims plus OneHot([0,1]) is the accepted selector; only the algebraically exact shared-basis final factorization remains changed.",
    }
    (HERE / "build_safe_argmax.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
