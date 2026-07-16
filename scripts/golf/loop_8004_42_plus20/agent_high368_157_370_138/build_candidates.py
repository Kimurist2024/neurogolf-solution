#!/usr/bin/env python3
"""Build exact/local re-golf probes for lane 138 without touching root artifacts."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE / "baseline"
CANDIDATES = HERE / "candidates"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
MEMBER_SHA256 = {
    157: "a1254f2619406b8db5d3fe5fdd1c42c917820fa51b91faef0f3ceed5d8b3662f",
    368: "0d950f5053aa62e7a3208be01514ad061b85580875e0e93aa7ee941cbacaa811",
    370: "513c0b40056f0ef9ee30cffe32a940571a0e977bf467d1c90096425c68e682d9",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXACT = load_module(
    "lane138_exact",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8008_exact_white102/scan_exact.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_uses(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new


def remove_nodes(model: onnx.ModelProto, outputs: set[str]) -> None:
    keep = [
        node for node in model.graph.node
        if not any(name in outputs for name in node.output if name)
    ]
    del model.graph.node[:]
    model.graph.node.extend(keep)


def remove_unused_initializers(model: onnx.ModelProto) -> list[str]:
    live = {name for node in model.graph.node for name in node.input if name}
    live.update(value.name for value in model.graph.input)
    live.update(value.name for value in model.graph.output)
    removed = [item.name for item in model.graph.initializer if item.name not in live]
    keep = [item for item in model.graph.initializer if item.name in live]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return removed


def castlike_to_cast(model: onnx.ModelProto) -> list[dict[str, Any]]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    values = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    initializer_types = {
        item.name: int(item.data_type) for item in inferred.graph.initializer
    }
    actions = []
    for node in model.graph.node:
        if node.op_type != "CastLike" or len(node.input) != 2:
            continue
        like = values.get(node.input[1])
        if like is None and node.input[1] not in initializer_types:
            continue
        if like is not None and not like.type.HasField("tensor_type"):
            continue
        target = (
            int(like.type.tensor_type.elem_type)
            if like is not None
            else initializer_types[node.input[1]]
        )
        actions.append(
            {
                "output": node.output[0],
                "dtype": onnx.TensorProto.DataType.Name(target),
                "proof": "CastLike second input supplies dtype only; Cast(to) is exact",
            }
        )
        node.op_type = "Cast"
        del node.input[1:]
        del node.attribute[:]
        node.attribute.append(helper.make_attribute("to", target))
    return actions


def qlinear_default_zero_points(model: onnx.ModelProto) -> list[dict[str, Any]]:
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    actions = []
    for node in model.graph.node:
        if node.op_type != "QLinearConv" or len(node.input) < 8:
            continue
        for index, label in ((2, "x_zero_point"), (5, "w_zero_point"), (7, "y_zero_point")):
            name = node.input[index]
            array = arrays.get(name)
            if array is None or not np.all(array == 0):
                continue
            node.input[index] = ""
            actions.append(
                {
                    "node": node.output[0],
                    "input": label,
                    "initializer": name,
                    "proof": "QLinearConv omitted zero-point defaults to numeric zero of matching type",
                }
            )
    return actions


def fold_fixed_shapes(
    model: onnx.ModelProto,
    outputs: tuple[str, ...],
) -> list[dict[str, Any]]:
    fixed = {
        "batch_shape": [1],
        "h_shape30": [30],
        "h_shape29": [29],
        "h_shape28": [28],
        "h_shape27": [27],
        "h_shape26": [26],
        "h_shape25": [25],
        "h_shape24": [24],
        "h_shape23": [23],
        "h_shape22": [22],
        "h_shape21": [21],
        "h_shape20": [20],
        "shape10": [10],
    }
    chosen = set(outputs)
    actions = []
    for name in outputs:
        if name not in fixed:
            raise KeyError(name)
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(fixed[name], dtype=np.int64), name)
        )
        actions.append(
            {
                "output": name,
                "value": fixed[name],
                "proof": "fixed canonical input/derived tensor shape contract",
            }
        )
    remove_nodes(model, chosen)
    return actions


def bypass_ccp(model: onnx.ModelProto, output: str) -> dict[str, Any]:
    node = next(node for node in model.graph.node if output in node.output)
    if node.op_type != "CenterCropPad":
        raise ValueError(f"{output} is {node.op_type}")
    source = node.input[0]
    replace_uses(model, output, source)
    remove_nodes(model, {output})
    return {
        "output": output,
        "source": source,
        "proof": "CenterCropPad target equals source extent on the selected axis",
    }


def prelu_neg_to_neg(model: onnx.ModelProto) -> list[dict[str, Any]]:
    node = next(node for node in model.graph.node if "neg_mask3" in node.output)
    if node.op_type != "PRelu" or node.input[0] != "neg_one_1111":
        raise ValueError("unexpected neg_mask3 producer")
    node.op_type = "Neg"
    del node.input[:]
    node.input.append("mask3_pos")
    del node.attribute[:]
    removed = remove_unused_initializers(model)
    return [
        {
            "output": "neg_mask3",
            "proof": "PRelu(-1,slope) equals -slope for finite nonnegative slope",
            "removed_initializers": removed,
        }
    ]


def prelu_pair_to_mul(model: onnx.ModelProto) -> list[dict[str, Any]]:
    prelu = next(node for node in model.graph.node if "prelu_prod" in node.output)
    if prelu.op_type != "PRelu":
        raise ValueError("unexpected prelu_prod producer")
    prelu.op_type = "Mul"
    del prelu.input[:]
    prelu.input.extend(["mask3_pos", "selector3_slope"])
    remove_nodes(model, {"neg_mask3"})
    removed = remove_unused_initializers(model)
    return [
        {
            "removed": "neg_mask3",
            "output": "prelu_prod",
            "proof": "ReduceL1(abs((-mask)*selector)) equals ReduceL1(mask*selector) for nonnegative masks/selectors",
            "removed_initializers": removed,
        }
    ]


def prelu_base_to_mul(model: onnx.ModelProto) -> list[dict[str, Any]]:
    node = next(node for node in model.graph.node if "neg_kernel_base_pair" in node.output)
    if node.op_type != "PRelu":
        raise ValueError("unexpected kernel PRelu")
    node.op_type = "Mul"
    del node.attribute[:]
    return [
        {
            "output": "neg_kernel_base_pair",
            "proof": "all base_pair_neg elements are strictly negative, so PRelu(x,s)=x*s",
        }
    ]


def replace_node_block(
    model: onnx.ModelProto,
    removed_outputs: set[str],
    new_nodes: list[onnx.NodeProto],
) -> None:
    indices = [
        index
        for index, node in enumerate(model.graph.node)
        if any(name in removed_outputs for name in node.output if name)
    ]
    if not indices:
        raise ValueError(f"node block not found: {sorted(removed_outputs)}")
    insert_at = min(indices)
    keep = [
        node
        for node in model.graph.node
        if not any(name in removed_outputs for name in node.output if name)
    ]
    del model.graph.node[:]
    model.graph.node.extend(keep[:insert_at])
    model.graph.node.extend(new_nodes)
    model.graph.node.extend(keep[insert_at:])


def bf3_sub_chain(model: onnx.ModelProto) -> list[dict[str, Any]]:
    removed = {
        "bf3_used01",
        "bf3_used012",
        "bf3_unused_mask",
        "bf3_unused_neg",
        "bf3_raw",
    }
    nodes = [
        helper.make_node("Sub", ["bstarts_s", "blue_factor_0"], ["bf3_sub0"], name="bf3_sub0"),
        helper.make_node("Sub", ["bf3_sub0", "blue_factor_1"], ["bf3_sub1"], name="bf3_sub1"),
        helper.make_node("Sub", ["bf3_sub1", "blue_factor_2"], ["bf3_raw"], name="bf3_raw"),
    ]
    replace_node_block(model, removed, nodes)
    return [
        {
            "removed_outputs": sorted(removed),
            "new_outputs": ["bf3_sub0", "bf3_sub1", "bf3_raw"],
            "proof": "generator guarantees unique footprints, hence three selected factors are distinct bits in bstarts_s; bitset subtraction leaves exactly the fourth factor",
        }
    ]


def bf3_sum_sub(model: onnx.ModelProto) -> list[dict[str, Any]]:
    removed = {
        "bf3_used01",
        "bf3_used012",
        "bf3_unused_mask",
        "bf3_unused_neg",
        "bf3_raw",
    }
    nodes = [
        helper.make_node(
            "Sum",
            ["blue_factor_0", "blue_factor_1", "blue_factor_2"],
            ["bf3_used_sum"],
            name="bf3_used_sum",
        ),
        helper.make_node("Sub", ["bstarts_s", "bf3_used_sum"], ["bf3_raw"], name="bf3_raw"),
    ]
    replace_node_block(model, removed, nodes)
    return [
        {
            "removed_outputs": sorted(removed),
            "new_outputs": ["bf3_used_sum", "bf3_raw"],
            "proof": "distinct one-bit selected factors have arithmetic sum equal to bitwise union",
        }
    ]


def row_variadic_sum(model: onnx.ModelProto) -> list[dict[str, Any]]:
    actions = []
    for row, sources in (
        ("bo3_all", [f"row3_{i}" for i in range(4)]),
        ("bo4_all", [f"row4_{i}" for i in range(4)]),
        ("bo5_all", [f"r2s3_sub_{i}" for i in range(4)]),
    ):
        if row == "bo3_all":
            removed = {"bo3_01", "bo3_23", "bo3_all"}
        elif row == "bo4_all":
            removed = {"bo4_01", "bo4_23", "bo4_all"}
        else:
            removed = {"bo5_01", "bo5_23", "bo5_all"}
        replace_node_block(
            model,
            removed,
            [helper.make_node("Sum", sources, [row], name=row)],
        )
        actions.append(
            {
                "row": row,
                "sources": sources,
                "proof": "generator's one-cell-margin non-overlap makes source bitfields disjoint, so Sum equals BitwiseOr",
            }
        )
    return actions


def delta_where_u16(model: onnx.ModelProto) -> list[dict[str, Any]]:
    replacements = {
        "delta_train_num_f16": np.asarray([1025], dtype=np.uint16),
        "delta_a117_num_f16": np.asarray([64], dtype=np.uint16),
        "delta_a187_num_f16": np.asarray([512], dtype=np.uint16),
    }
    kept = [item for item in model.graph.initializer if item.name not in replacements]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    for name, array in replacements.items():
        model.graph.initializer.append(numpy_helper.from_array(array, name))
    nodes = list(model.graph.node)
    cast_index = next(
        index for index, node in enumerate(nodes) if "delta_u16" in node.output
    )
    cast_node = nodes[cast_index]
    if cast_node.op_type != "Cast" or cast_node.input != ["delta_f16"]:
        raise ValueError("unexpected delta Cast")
    where1 = next(node for node in nodes if "delta_f16_1" in node.output)
    where2 = next(node for node in nodes if "delta_f16_2" in node.output)
    where3 = next(node for node in nodes if "delta_f16" in node.output)
    where1.input[2] = "zero16"
    where3.output[0] = "delta_u16"
    del model.graph.node[cast_index]
    removed = remove_unused_initializers(model)
    return [
        {
            "outputs": ["delta_f16_1", "delta_f16_2", "delta_u16"],
            "proof": "all four branch constants are exactly representable uint16 integers; Where is value-preserving and the terminal Cast becomes identity",
            "removed_initializers": removed,
        }
    ]


def combined(model: onnx.ModelProto, functions: list[Callable[[onnx.ModelProto], Any]]) -> list[Any]:
    return [function(model) for function in functions]


def add_variant(
    rows: list[dict[str, Any]],
    seen: set[tuple[int, str]],
    task: int,
    kind: str,
    base: onnx.ModelProto,
    transform: Callable[[onnx.ModelProto], Any],
    theorem: str,
) -> None:
    model = copy.deepcopy(base)
    actions = transform(model)
    data = model.SerializeToString()
    digest = sha256(data)
    if digest == MEMBER_SHA256[task] or (task, digest) in seen:
        return
    seen.add((task, digest))
    path = CANDIDATES / f"task{task:03d}_{kind}_{digest[:12]}.onnx"
    path.write_bytes(data)
    rows.append(
        {
            "task": task,
            "kind": kind,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
            "authority_sha256": MEMBER_SHA256[task],
            "theorem": theorem,
            "actions": actions,
        }
    )


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    if sha256((ROOT / "submission.zip").read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority ZIP drift")
    bases = {}
    for task in MEMBER_SHA256:
        path = BASE / f"task{task:03d}.onnx"
        data = path.read_bytes()
        if sha256(data) != MEMBER_SHA256[task]:
            raise RuntimeError(f"task{task:03d} baseline drift")
        bases[task] = onnx.load_model_from_string(data)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    # Generic all-input structure transforms. Only normalization changes these incumbents.
    for task, base in bases.items():
        for kind in EXACT.KINDS:
            candidate, actions = EXACT.transform(base, kind)
            if not actions.get("semantic_action_count") and not actions.get("metadata_action_count"):
                continue
            add_variant(
                rows,
                seen,
                task,
                f"generic_{kind}",
                base,
                lambda model, candidate=candidate: (
                    model.CopyFrom(candidate), actions
                )[1],
                "generic exact transform from lane102; metadata normalization alone is not promotable",
            )

    add_variant(
        rows, seen, 157, "bf3_sub_chain", bases[157], bf3_sub_chain,
        "generator-proved unique selected factors permit exact set subtraction",
    )
    add_variant(
        rows, seen, 157, "bf3_sum_sub", bases[157], bf3_sum_sub,
        "generator-proved unique selected factors permit exact arithmetic union/subtraction",
    )
    add_variant(
        rows, seen, 157, "row_variadic_sum", bases[157], row_variadic_sum,
        "generator-proved non-overlap makes row bitfields arithmetically disjoint",
    )
    add_variant(
        rows, seen, 157, "bf3_sub_rows_sum", bases[157],
        lambda model: combined(model, [bf3_sub_chain, row_variadic_sum]),
        "composition of generator-proved distinct-factor and disjoint-row identities",
    )
    add_variant(
        rows, seen, 157, "bf3_sum_rows_sum", bases[157],
        lambda model: combined(model, [bf3_sum_sub, row_variadic_sum]),
        "composition of generator-proved arithmetic-union identities",
    )
    add_variant(
        rows, seen, 157, "delta_where_u16", bases[157], delta_where_u16,
        "full uint16-domain exact branch typing",
    )

    add_variant(
        rows, seen, 368, "cast_attribute", bases[368], castlike_to_cast,
        "CastLike-to-Cast preserves all tensor values and shapes",
    )
    add_variant(
        rows, seen, 368, "qlinear_default_zero", bases[368], qlinear_default_zero_points,
        "schema-defined default zero-points are exact",
    )
    add_variant(
        rows, seen, 368, "qlinear_zero_plus_cast", bases[368],
        lambda model: combined(model, [castlike_to_cast, qlinear_default_zero_points, remove_unused_initializers]),
        "schema-defined zero defaults plus exact Cast attributeization",
    )

    chain = tuple(["h_shape30", "batch_shape"] + [f"h_shape{i}" for i in range(29, 19, -1)])
    add_variant(
        rows, seen, 370, "fold_batch_shape", bases[370],
        lambda model: fold_fixed_shapes(model, ("batch_shape",)),
        "input batch dimension is statically fixed to one",
    )
    add_variant(
        rows, seen, 370, "fold_shape10", bases[370],
        lambda model: fold_fixed_shapes(model, ("shape10",)),
        "counts has statically fixed shape [10]",
    )
    add_variant(
        rows, seen, 370, "fold_all_height_shapes", bases[370],
        lambda model: fold_fixed_shapes(model, chain),
        "canonical input height is 30 and the Sub chain is exact integer decrement",
    )
    add_variant(
        rows, seen, 370, "fold_all_shapes", bases[370],
        lambda model: fold_fixed_shapes(model, chain + ("shape10",)),
        "all Shape/Sub values are fixed by the canonical contract",
    )
    for output in ("pow_f_hid", "base_pair_hid", "selector3_b"):
        add_variant(
            rows, seen, 370, f"bypass_{output}", bases[370],
            lambda model, output=output: bypass_ccp(model, output),
            "CenterCropPad is identity on its selected fixed-size axis",
        )
    add_variant(
        rows, seen, 370, "bypass_three_ccp", bases[370],
        lambda model: [bypass_ccp(model, name) for name in ("pow_f_hid", "base_pair_hid", "selector3_b")],
        "three independent CenterCropPad identities",
    )
    add_variant(rows, seen, 370, "cast_attributes", bases[370], castlike_to_cast,
                "CastLike-to-Cast preserves all values and shapes")
    add_variant(rows, seen, 370, "prelu_neg", bases[370], prelu_neg_to_neg,
                "PRelu negative scalar identity")
    add_variant(rows, seen, 370, "prelu_pair_mul", bases[370], prelu_pair_to_mul,
                "nonnegative selector/mask algebra under downstream ReduceL1")
    add_variant(rows, seen, 370, "prelu_base_mul", bases[370], prelu_base_to_mul,
                "PRelu on strictly-negative constant base equals Mul")
    add_variant(
        rows, seen, 370, "prelu_pair_fold_all_shapes", bases[370],
        lambda model: combined(model, [prelu_pair_to_mul, lambda m: fold_fixed_shapes(m, chain + ("shape10",))]),
        "pair algebra plus fixed shape contract",
    )
    add_variant(
        rows, seen, 370, "all_local_exact", bases[370],
        lambda model: combined(
            model,
            [
                castlike_to_cast,
                prelu_pair_to_mul,
                prelu_base_to_mul,
                lambda m: [bypass_ccp(m, name) for name in ("pow_f_hid", "base_pair_hid", "selector3_b")],
                lambda m: fold_fixed_shapes(m, chain + ("shape10",)),
                remove_unused_initializers,
            ],
        ),
        "composition of exact fixed-domain local identities",
    )

    # task157 is already the 4-byte factored form. Re-emit generic normalization only;
    # no new private fixture or approximation is introduced.
    manifest = {
        "authority_zip_sha256": AUTHORITY_SHA256,
        "authority_member_sha256": MEMBER_SHA256,
        "candidate_count": len(rows),
        "rows": rows,
    }
    (HERE / "audit/build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "by_task": {str(t): sum(r["task"] == t for r in rows) for t in MEMBER_SHA256}}, indent=2))


if __name__ == "__main__":
    main()
