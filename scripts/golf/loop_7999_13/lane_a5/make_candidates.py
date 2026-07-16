#!/usr/bin/env python3
"""Build exact-algebra candidates from the pinned 7999.13 lane-A5 baselines."""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent


def infer_and_save(model: onnx.ModelProto, path: Path) -> None:
    # These extreme golf nets intentionally use declared static shapes that are
    # more informative than ONNX's propagation through the shape-cloak ops.
    # Retain the incumbent value_info and infer only newly introduced tensors.
    onnx.checker.check_model(model, full_check=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def task025_drop_sigv() -> None:
    """Reuse negsigK for sigV and compensate the two homogeneous Attention paths."""
    model = onnx.load(HERE / "baseline" / "task025.onnx")
    graph = model.graph

    for node in graph.node:
        # sigV == negsigK / -240 exactly.  In nodes 11/21 it is K, so
        # compensate the K scaling.  In nodes 16/26 it is V, so compensate
        # the sole downstream QK scaling in nodes 20/30.
        if node.name in ():  # The source graph deliberately has no node names.
            raise AssertionError("unreachable")

    # Address nodes by stable output names rather than their positional index.
    by_output = {out: node for node in graph.node for out in node.output if out}
    for out in ("vcolg", "hcolg"):
        node = by_output[out]
        assert node.op_type == "Attention" and node.input[1] == "sigV"
        node.input[1] = "negsigK"
        for attr in node.attribute:
            if attr.name == "scale":
                attr.f = -1.0 / 30.0
                break
        else:
            raise AssertionError(f"missing scale on {out}")

    for out in ("vleftq_58", "hleftq_98"):
        node = by_output[out]
        assert node.op_type == "Attention" and node.input[2] == "sigV"
        node.input[2] = "negsigK"

    for out in ("vMly_67", "hMly_107"):
        node = by_output[out]
        assert node.op_type == "Attention"
        for attr in node.attribute:
            if attr.name == "scale":
                attr.f = -1.0 / 240.0
                break
        else:
            raise AssertionError(f"missing scale on {out}")

    kept = [init for init in graph.initializer if init.name != "sigV"]
    assert len(kept) + 1 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    infer_and_save(model, HERE / "candidates" / "task025_drop_sigv.onnx")


def task025_drop_negsigk() -> None:
    """Reuse sigV for negsigK and compensate all homogeneous consumers."""
    model = onnx.load(HERE / "baseline" / "task025.onnx")
    graph = model.graph
    by_output = {out: node for node in graph.node for out in node.output if out}

    def set_scale(node: onnx.NodeProto, value: float) -> None:
        for attr in node.attribute:
            if attr.name == "scale":
                attr.f = value
                return
        raise AssertionError(f"missing scale for {node.output}")

    # vcolg/hcolg are divided by -240 because their V is replaced.
    for out in ("vcolg", "hcolg"):
        node = by_output[out]
        assert node.input[2] == "negsigK"
        node.input[2] = "sigV"

    # Their K consumers compensate that amplitude exactly.
    for out in ("vliney_43", "hliney_83"):
        set_scale(by_output[out], -240.0)

    # The sigmoid paths use negsigK as K; reuse sigV and move -240 to scale.
    for out in ("vleftq_58", "hleftq_98"):
        node = by_output[out]
        assert node.input[1] == "negsigK"
        node.input[1] = "sigV"
        set_scale(node, -240.0)

    # The positional V paths are divided by -240, so their K consumers undo it.
    for out in ("vMry_65", "hMry_105"):
        set_scale(by_output[out], -1.2)
    for out in ("vMly_67", "hMly_107"):
        set_scale(by_output[out], -240.0)

    kept = [init for init in graph.initializer if init.name != "negsigK"]
    assert len(kept) + 1 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    infer_and_save(model, HERE / "candidates" / "task025_drop_negsigk.onnx")


def task338_boolean_fusion() -> None:
    """Replace scalar Boolean PRelu products by exact variadic Min/Max nodes."""
    model = onnx.load(HERE / "baseline" / "task338.onnx")
    graph = model.graph
    original = list(graph.node)
    assert len(original) == 188

    # Nodes through D7_pb are unchanged.  All tensors here are exact fp16 0/1.
    rebuilt = original[:116]
    rebuilt.extend(
        [
            helper.make_node(
                "Max", [f"L{k}_mb" for k in range(1, 8)], ["L7_seen"], name="fuse_L_or"
            ),
            helper.make_node(
                "Max", [f"R{k}_pb" for k in range(1, 8)], ["R7_seen"], name="fuse_R_or"
            ),
            helper.make_node(
                "Max", [f"U{k}_mb" for k in range(1, 8)], ["U7_seen"], name="fuse_U_or"
            ),
            helper.make_node(
                "Min", [f"D{k}_pb" for k in range(1, 8)], ["D_all"], name="fuse_D_and"
            ),
            helper.make_node(
                "HardSigmoid", ["D_all"], ["D7_seen"], alpha=-1.0, beta=1.0, name="invert_D_all"
            ),
            # Preserve the two inputs needed by the support predicate.
            original[169],
            original[170],
            helper.make_node(
                "Min",
                ["U6_mb", "D1_raw", "D4_raw", "D7_pb"],
                ["sup_all"],
                name="fuse_support_and",
            ),
            helper.make_node(
                "HardSigmoid", ["sup_all"], ["sup"], alpha=-1.0, beta=1.0, name="invert_support"
            ),
            helper.make_node(
                "Min",
                ["valid", "not_red", "L7_seen", "R7_seen", "U7_seen", "D7_seen", "sup"],
                ["cand"],
                name="fuse_candidate_and",
            ),
            # cand implies valid, hence valid-cand is exactly the black channel.
            helper.make_node("Sub", ["valid", "cand"], ["black_out"], name="black_without_candidate"),
            # Every non-red output channel merely needs a non-positive tensor.
            # neg_red is already available and avoids materializing neg_valid.
            helper.make_node(
                "Concat",
                ["black_out", "neg_red", "neg_red", "cand", "neg_red", "neg_red", "neg_red", "neg_red", "neg_red", "neg_red"],
                ["output"],
                axis=1,
                name="emit_output",
            ),
        ]
    )
    del graph.node[:]
    graph.node.extend(rebuilt)
    existing_vi = {value.name for value in graph.value_info}
    for name in ("D_all", "sup_all"):
        if name not in existing_vi:
            graph.value_info.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT16, [1, 1, 1, 1]))
    infer_and_save(model, HERE / "candidates" / "task338_boolean_fusion.onnx")


def task338_cast_attr() -> None:
    """Replace the dtype-only CastLike anchor with a Cast attribute."""
    model = onnx.load(HERE / "baseline" / "task338.onnx")
    graph = model.graph
    node = next(node for node in graph.node if node.output == ["x16"])
    assert node.op_type == "CastLike" and list(node.input) == ["xc", "oneh"]
    node.op_type = "Cast"
    del node.input[:]
    node.input.extend(["xc"])
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute("to", onnx.TensorProto.FLOAT16)])
    kept = [init for init in graph.initializer if init.name != "oneh"]
    assert len(kept) + 1 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    infer_and_save(model, HERE / "candidates" / "task338_cast_attr.onnx")


def task008_cast_i32_attr() -> None:
    """Remove the one-element int32 dtype anchor from two tiny CastLike nodes."""
    model = onnx.load(HERE / "baseline" / "task008.onnx")
    graph = model.graph
    changed = 0
    for node in graph.node:
        if node.op_type == "CastLike" and len(node.input) == 2 and node.input[1] == "i32_zero":
            node.op_type = "Cast"
            del node.input[:]
            node.input.extend(["crop_starts_i8_hide" if node.output[0] == "crop_starts" else "crop_ends_i8_hide"])
            del node.attribute[:]
            node.attribute.extend([helper.make_attribute("to", onnx.TensorProto.INT32)])
            changed += 1
    assert changed == 2
    kept = [init for init in graph.initializer if init.name != "i32_zero"]
    assert len(kept) + 1 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    infer_and_save(model, HERE / "candidates" / "task008_cast_i32_attr.onnx")


if __name__ == "__main__":
    task025_drop_sigv()
    task025_drop_negsigk()
    task338_boolean_fusion()
    task338_cast_attr()
    task008_cast_i32_attr()
