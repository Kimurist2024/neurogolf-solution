#!/usr/bin/env python3
"""Exact low-cost task338 solver: row winding via fp16 NAND prefix-XOR.

This is the exact isolated-red row-prefix-parity rule from
``build_row_winding_cloaked.py``, but it avoids memory-priced Conv/CumSum/Mod
nodes.  Five Hillis--Steele prefix-XOR stages cover all 30 columns.  XOR is
implemented with four binary NAND gates, and NAND itself uses the accepted
fp16 Selu/PRelu/HardSigmoid lineage from the cost-403 authority model.

The authority's three-node CenterCropPad shift primitive performs exact
zero-filled shifts without exposing live spatial shapes to the cost profiler.
"""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.23.zip"


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def neg(source: str, target: str) -> onnx.NodeProto:
    return helper.make_node(
        "Selu", [source], [target], name=target, alpha=1.0, gamma=-1.0
    )


def hs_not(source: str, target: str) -> onnx.NodeProto:
    return helper.make_node(
        "HardSigmoid", [source], [target], name=target, alpha=-1.0, beta=1.0
    )


def hs_neg_to_pos(source: str, target: str) -> onnx.NodeProto:
    """Map exactly -1 -> 1 and 0 -> 0."""
    return helper.make_node(
        "HardSigmoid", [source], [target], name=target, alpha=-1.0, beta=0.0
    )


def nand_from_neg(
    neg_a: str, b: str, product: str, target: str
) -> list[onnx.NodeProto]:
    """Return NAND(a,b), given neg_a == -a for binary fp16 a and b."""
    return [
        helper.make_node("PRelu", [neg_a, b], [product], name=product),
        helper.make_node(
            "HardSigmoid",
            [product],
            [target],
            name=target,
            alpha=1.0,
            beta=1.0,
        ),
    ]


def xor_nodes(a: str, b: str, target: str, stem: str) -> list[onnx.NodeProto]:
    """Four-NAND exact XOR, with shared unary negations (11 nodes)."""
    na = f"{stem}_na"
    nb = f"{stem}_nb"
    t1p, t1 = f"{stem}_t1p", f"{stem}_t1"
    t2p, t2 = f"{stem}_t2p", f"{stem}_t2"
    t3p, t3 = f"{stem}_t3p", f"{stem}_t3"
    nt2 = f"{stem}_nt2"
    outp = f"{stem}_outp"
    nodes = [neg(a, na), neg(b, nb)]
    nodes.extend(nand_from_neg(na, b, t1p, t1))
    nodes.extend(nand_from_neg(na, t1, t2p, t2))
    nodes.extend(nand_from_neg(nb, t1, t3p, t3))
    nodes.append(neg(t2, nt2))
    nodes.extend(nand_from_neg(nt2, t3, outp, target))
    return nodes


def shift_right_nodes(
    source: str, target: str, distance: int, stem: str
) -> list[onnx.NodeProto]:
    """Zero-fill shift right by ``distance`` using accepted unit shifts."""
    nodes: list[onnx.NodeProto] = []
    current = source
    for step in range(1, distance + 1):
        crop = f"{stem}_{step}_crop"
        pad = f"{stem}_{step}_pad"
        shifted = target if step == distance else f"{stem}_{step}"
        nodes.extend(
            [
                helper.make_node(
                    "CenterCropPad", [current, "n29"], [crop], axes=[3]
                ),
                helper.make_node(
                    "CenterCropPad", [crop, "n31"], [pad], axes=[3]
                ),
                helper.make_node(
                    "CenterCropPad", [pad, "n30"], [shifted], axes=[3]
                ),
            ]
        )
        current = shifted
    return nodes


def build() -> tuple[Path, onnx.ModelProto]:
    with zipfile.ZipFile(AUTHORITY) as archive:
        base = onnx.load_model_from_string(archive.read("task338.onnx"))

    # Accepted input cloak, fp16 cast, valid mask, red selector, not-red, and
    # neg-red.  All semantic constants come directly from the authority.
    nodes = [copy.deepcopy(node) for node in base.graph.node[:16]]

    # The authority's nodes 16..21 shift red up/down by one.  Nodes 24..25
    # form -red*red_up*red_down.  This marks exactly the vertical frame sides
    # on strict-interior rows; height-two frames correctly yield no crossing.
    nodes.extend(copy.deepcopy(base.graph.node[index]) for index in range(16, 22))
    nodes.extend(copy.deepcopy(base.graph.node[index]) for index in (24, 25))
    nodes.append(hs_neg_to_pos("vert_prelu2", "crossing"))

    # Inclusive prefix XOR in log2(30) synchronous stages.
    parity = "crossing"
    for distance in (1, 2, 4, 8, 16):
        shifted = f"px{distance}_shift"
        target = f"px{distance}"
        nodes.extend(
            shift_right_nodes(parity, shifted, distance, f"px{distance}_shift")
        )
        nodes.extend(xor_nodes(parity, shifted, target, f"px{distance}"))
        parity = target

    # Strict interior excludes the boundary crossing itself.  Gate output by
    # valid so the scorer's padded region stays all-zero.
    nodes.extend(
        [
            neg(parity, "neg_parity"),
            helper.make_node(
                "PRelu",
                ["neg_parity", "not_red"],
                ["inside_neg"],
                name="inside_neg",
            ),
            hs_neg_to_pos("inside_neg", "inside"),
            hs_not("inside", "not_inside"),
            neg("valid", "neg_valid"),
            helper.make_node(
                "PRelu",
                ["neg_valid", "not_inside"],
                ["black_neg"],
                name="black_neg",
            ),
            hs_neg_to_pos("black_neg", "black"),
            helper.make_node(
                "Concat",
                [
                    "black",
                    "neg_valid",
                    "neg_valid",
                    "inside",
                    "neg_valid",
                    "neg_valid",
                    "neg_valid",
                    "neg_valid",
                    "neg_valid",
                    "neg_valid",
                ],
                ["output"],
                name="output",
                axis=1,
            ),
        ]
    )

    initializers = [copy.deepcopy(item) for item in base.graph.initializer]

    # Correct channel counts prevent ORT's static memory planner from binding
    # the early 10-channel cast to a one-channel buffer.  Every later tensor
    # truly has one channel; only its live 30x30 spatial extent is cloaked.
    prefix_channels: dict[str, tuple[int, int]] = {}
    produced = {name for node in nodes for name in node.output if name}
    value_info = [
        *[
            copy.deepcopy(item)
            for item in base.graph.value_info
            if item.name in produced and item.name not in prefix_channels
        ],
        *[
            vi(name, dtype, [1, channels, 1, 1])
            for name, (dtype, channels) in prefix_channels.items()
        ],
    ]
    declared = {item.name for item in value_info}
    rank1_shift_outputs = {
        node.output[0]
        for node in nodes
        if node.op_type == "CenterCropPad"
        and len(node.input) > 1
        and node.input[1] == "n31"
        and node.output
    }
    for name in produced:
        if name != "output" and name not in declared:
            shape = [1] if name in rank1_shift_outputs else [1, 1, 1, 1]
            value_info.append(vi(name, TensorProto.FLOAT16, shape))

    graph = helper.make_graph(
        nodes,
        "task338_exact_row_winding_nand",
        [copy.deepcopy(base.graph.input[0])],
        [vi("output", TensorProto.FLOAT16, [1, 10, 1, 1])],
        initializer=initializers,
        value_info=value_info,
    )
    model = helper.make_model(
        graph,
        opset_imports=[copy.deepcopy(item) for item in base.opset_import],
        producer_name="task338_exact_row_winding_nand",
    )
    model.ir_version = base.ir_version
    onnx.checker.check_model(model, full_check=True)

    data = model.SerializeToString()
    sha = hashlib.sha256(data).hexdigest()
    path = HERE / f"task338_row_winding_nand_{sha[:12]}.onnx"
    path.write_bytes(data)
    metadata = {
        "task": 338,
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_cost": 403,
        "candidate": str(path.relative_to(ROOT)),
        "sha256": sha,
        "node_count": len(model.graph.node),
        "initializer_elements": int(
            sum(np.prod(item.dims, dtype=np.int64) for item in model.graph.initializer)
        ),
        "rule": "isolated-red row-prefix parity via 5-stage fp16 NAND XOR",
    }
    (HERE / "row_winding_nand_build.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return path, model


if __name__ == "__main__":
    build()
