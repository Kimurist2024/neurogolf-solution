#!/usr/bin/env python3
"""Build the exact task192 shared-basis/Hardmax rewrite.

The accepted polynomial model computes two row masks and two output bases with
separate tensors.  This rewrite stores only nonzero/background basis rows,
creates the selected-color row with Hardmax, and shares the resulting 3x10
basis in the final Einsum.  Small 2x3 coefficient matrices recover exactly the
old center, neighbor, and output factors.
"""

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
SOURCE = ROOT / "others/71407/task192.onnx"
OUTPUT = HERE / "candidates/task192_shared_basis_hardmax.onnx"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def main() -> None:
    source = onnx.load(SOURCE)
    adj = next(item for item in source.graph.initializer if item.name == "adj")

    nonzero = numpy_helper.from_array(
        np.asarray([[0.0] + [1.0] * 9], dtype=np.float32), "nonzero"
    )
    background = numpy_helper.from_array(
        np.asarray([[1.0] + [0.0] * 9], dtype=np.float32), "background"
    )
    center_map = numpy_helper.from_array(
        np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        "center_map",
    )
    neighbor_map = numpy_helper.from_array(
        np.asarray([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "neighbor_map",
    )
    route_out = numpy_helper.from_array(
        np.asarray([[0.0, 1.0, 0.0], [0.0, -9.0, 1.0]], dtype=np.float32),
        "route_out",
    )

    nodes = [
        helper.make_node(
            "Einsum",
            ["input", "nonzero"],
            ["hist"],
            equation="bchw,xc->xc",
            name="hist",
        ),
        helper.make_node("Hardmax", ["hist"], ["selected"], axis=1, name="selected"),
        helper.make_node(
            "Concat",
            ["nonzero", "background", "selected"],
            ["basis"],
            axis=0,
            name="basis",
        ),
        helper.make_node(
            "Einsum",
            [
                "input",
                "basis",
                "center_map",
                "input",
                "basis",
                "neighbor_map",
                "adj",
                "input",
                "basis",
                "neighbor_map",
                "adj",
                "route_out",
                "basis",
            ],
            ["output"],
            equation=(
                "bchw,kc,rk,bdhq,ld,rl,qw,bepw,me,rm,ph,rn,no->bohw"
            ),
            name="output",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task192_shared_basis_hardmax",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [adj, nonzero, background, center_map, neighbor_map, route_out],
        value_info=[
            helper.make_tensor_value_info("hist", TensorProto.FLOAT, [1, 10]),
            helper.make_tensor_value_info("selected", TensorProto.FLOAT, [1, 10]),
            helper.make_tensor_value_info("basis", TensorProto.FLOAT, [3, 10]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="codex-task192-basis178",
        ir_version=source.ir_version,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "source_profile": profile(SOURCE),
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "candidate_sha256": sha256(OUTPUT),
        "candidate_profile": profile(OUTPUT),
        "proof": {
            "basis_rows": ["nonzero", "background", "selected"],
            "center_map": "rows recover [inside, nonzero]",
            "neighbor_map": "rows recover [inside, selected]",
            "route_out": "rows recover [background, -9*background+selected]",
            "hardmax": "for hist shape [1,10], axis=1 equals ArgMax(first tie)+OneHot([0,1])",
            "final": "B*background + P*(-9*background+selected), identical to accepted polynomial",
        },
    }
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
