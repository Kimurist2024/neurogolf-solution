#!/usr/bin/env python3
"""Build exact task192 controls for the local-rule cost-floor audit."""

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
SOURCE = ROOT / "others/71407/FALLBACK_EXACT_DO_NOT_AUTO_MERGE/task192_exact1149.onnx.fallback"
POLICY_SOURCE = ROOT / "others/71407/task192.onnx"
CANDIDATES = HERE / "candidates"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name)


def build(output: Path, hardmax: bool) -> None:
    source = onnx.load(SOURCE)
    adjacency = next(x for x in source.graph.initializer if x.name == "adj")

    # Rows are inside-grid and nonzero. This same initializer is consumed
    # directly by the center factor. The second row is selected for the
    # histogram using OneHot's existing [0,1] values initializer.
    center_basis = tensor(
        "center_basis",
        np.asarray([[1.0] * 10, [0.0] + [1.0] * 9], dtype=np.float32),
    )
    neighbor_map = tensor(
        "neighbor_map",
        np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    # Relative to basis [inside, nonzero, selected]:
    # background = inside - nonzero;
    # predicate route = -9*background + selected.
    route_out = tensor(
        "route_out",
        np.asarray([[1.0, -1.0, 0.0], [-9.0, 9.0, 1.0]], dtype=np.float32),
    )
    onehot_values = tensor("onehot_values", np.asarray([0.0, 1.0], dtype=np.float32))
    initializers = [adjacency, center_basis, neighbor_map, route_out]
    if hardmax:
        # Keep a leading singleton so Hardmax itself returns rank 2. This is
        # only a rejected cost control; the accepted ArgMax path below reuses
        # onehot_values and avoids this duplicate two-element selector.
        initializers.append(
            tensor("hist_selector", np.asarray([[0.0, 1.0]], dtype=np.float32))
        )
        hist_inputs = ["input", "center_basis", "hist_selector"]
        hist_equation = "bchw,rc,xr->xc"
    else:
        initializers.append(onehot_values)
        hist_inputs = ["input", "center_basis", "onehot_values"]
        hist_equation = "bchw,rc,r->c"
    nodes = [
        helper.make_node(
            "Einsum", hist_inputs, ["hist"], equation=hist_equation, name="hist"
        )
    ]
    if hardmax:
        nodes.append(helper.make_node("Hardmax", ["hist"], ["selected"], axis=0, name="selected"))
    else:
        initializers.append(tensor("depth", np.asarray(10, dtype=np.int64)))
        nodes.extend([
            helper.make_node(
                "ArgMax", ["hist"], ["selected_i64"], axis=0, keepdims=1,
                name="selected_i64",
            ),
            helper.make_node(
                "OneHot", ["selected_i64", "depth", "onehot_values"], ["selected"],
                axis=-1, name="selected",
            ),
        ])
    nodes.extend([
        helper.make_node(
            "Concat", ["center_basis", "selected"], ["basis"], axis=0, name="basis"
        ),
        helper.make_node(
            "Einsum",
            [
                "input", "center_basis",
                "input", "basis", "neighbor_map", "adj",
                "input", "basis", "neighbor_map", "adj",
                "route_out", "basis",
            ],
            ["output"],
            equation="bchw,rc,bdhq,ld,rl,qw,bepw,me,rm,ph,rn,no->bohw",
            name="output",
        ),
    ])
    graph = helper.make_graph(
        nodes,
        "task192_center_direct_" + ("hardmax" if hardmax else "argmax"),
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="codex-task192-local211",
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def build_policy_pass_through(output: Path) -> None:
    """Apply only the center-basis identity to the staged POLICY90 graph."""
    source = onnx.load(POLICY_SOURCE)
    adjacency = next(x for x in source.graph.initializer if x.name == "adj")
    center_basis = tensor(
        "center_basis",
        np.asarray([[1.0] * 10, [0.0] + [1.0] * 9], dtype=np.float32),
    )
    hist_selector = tensor(
        "hist_selector", np.asarray([[0.0, 1.0]], dtype=np.float32)
    )
    neighbor_map = tensor(
        "neighbor_map",
        np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    route_out = tensor(
        "route_out",
        np.asarray([[1.0, -1.0, 0.0], [-9.0, 9.0, 1.0]], dtype=np.float32),
    )
    nodes = [
        helper.make_node(
            "Einsum",
            ["input", "center_basis", "hist_selector"],
            ["hist"],
            equation="bchw,rc,xr->xc",
            name="hist",
        ),
        helper.make_node(
            "HardSigmoid", ["hist"], ["selected"],
            alpha=1.0, beta=-33.0, name="selected_count_gt_33",
        ),
        helper.make_node(
            "Concat", ["center_basis", "selected"], ["basis"], axis=0, name="basis"
        ),
        helper.make_node(
            "Einsum",
            [
                "input", "center_basis",
                "input", "basis", "neighbor_map", "adj",
                "input", "basis", "neighbor_map", "adj",
                "route_out", "basis",
            ],
            ["output"],
            equation="bchw,rc,bdhq,ld,rl,qw,bepw,me,rm,ph,rn,no->bohw",
            name="output",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task192_policy90_center_direct_pass_through",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [adjacency, center_basis, hist_selector, neighbor_map, route_out],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="codex-task192-local211-policy90-pass-through",
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def main() -> None:
    exact = CANDIDATES / "task192_center_direct_argmax.onnx"
    rejected_control = CANDIDATES / "task192_center_direct_hardmax_rejected.onnx"
    policy = CANDIDATES / "task192_policy90_center_direct.onnx"
    build(exact, hardmax=False)
    build(rejected_control, hardmax=True)
    build_policy_pass_through(policy)
    result = {
        "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE), "profile": profile(SOURCE)},
        "exact_argmax": {"path": str(exact.relative_to(ROOT)), "sha256": sha256(exact), "profile": profile(exact)},
        "hardmax_rejected_control": {
            "path": str(rejected_control.relative_to(ROOT)),
            "sha256": sha256(rejected_control),
            "profile": profile(rejected_control),
            "rejected": "SOUND_REBUILD_PROMPT bans Hardmax even though this is an algebraic control",
        },
        "policy90_pass_through": {
            "source_path": str(POLICY_SOURCE.relative_to(ROOT)),
            "source_sha256": sha256(POLICY_SOURCE),
            "source_profile": profile(POLICY_SOURCE),
            "path": str(policy.relative_to(ROOT)),
            "sha256": sha256(policy),
            "profile": profile(policy),
            "classification": "POLICY90_INHERITED_RAW_PASS_THROUGH_CANDIDATE",
        },
    }
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
