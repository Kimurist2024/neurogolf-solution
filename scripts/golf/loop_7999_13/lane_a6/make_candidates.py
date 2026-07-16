#!/usr/bin/env python3
"""Build exact-algebra candidates from the pinned 7999.13 A6 members."""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def save_checked(model: onnx.ModelProto, name: str) -> None:
    onnx.checker.check_model(model, full_check=True)
    path = HERE / "candidates" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def task037_direct_shape_vector() -> None:
    """Store [29,31] directly instead of computing it with int64 PRelu."""
    model = onnx.load(HERE / "baseline" / "task037.onnx")
    graph = model.graph
    assert graph.node[5].op_type == "PRelu" and list(graph.node[5].output) == ["s29_31"]
    del graph.node[5]

    kept = [init for init in graph.initializer if init.name not in {"neg1_i64", "mix_29_31"}]
    kept.append(numpy_helper.from_array(np.asarray([29, 31], dtype=np.int64), name="s29_31"))
    assert len(kept) + 1 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    # The baseline deliberately has a rank-0 value_info for this rank-1
    # PRelu result.  A direct initializer cannot retain that shape cloak.
    value_info = [vi for vi in graph.value_info if vi.name != "s29_31"]
    del graph.value_info[:]
    graph.value_info.extend(value_info)
    save_checked(model, "task037_direct_shape_vector.onnx")


def task048_neg_and_bool_cast() -> None:
    """Eliminate the scalar zero if uint8 Neg/Cast are supported by ORT."""
    model = onnx.load(HERE / "baseline" / "task048.onnx")
    graph = model.graph
    assert graph.node[9].op_type == "Sub"
    assert list(graph.node[9].input) == ["zero_u8", "selected_redrow"]
    graph.node[9].op_type = "Neg"
    del graph.node[9].input[:]
    graph.node[9].input.extend(["selected_redrow"])

    assert graph.node[78].op_type == "Greater"
    assert list(graph.node[78].input) == ["hm", "zero_u8"]
    graph.node[78].op_type = "Cast"
    del graph.node[78].input[:]
    graph.node[78].input.extend(["hm"])
    graph.node[78].attribute.extend([onnx.helper.make_attribute("to", onnx.TensorProto.BOOL)])

    kept = [init for init in graph.initializer if init.name != "zero_u8"]
    assert len(kept) + 1 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    save_checked(model, "task048_neg_and_bool_cast.onnx")


def task048_not_sub_and_bool_cast() -> None:
    """Derive -x mod 256 as ~(x-1), reusing the existing uint8 one."""
    model = onnx.load(HERE / "baseline" / "task048.onnx")
    graph = model.graph
    neg = next(node for node in graph.node if "neg_selected_redrow" in node.output)
    assert neg.op_type == "Sub" and list(neg.input) == ["zero_u8", "selected_redrow"]
    del neg.input[:]
    neg.input.extend(["selected_redrow", "one_u8"])
    neg.output[0] = "selected_minus_one"
    neg.name = "selected_minus_one"
    neg_index = next(i for i, node in enumerate(graph.node) if node is neg)
    bitnot = onnx.helper.make_node(
        "BitwiseNot", ["selected_minus_one"], ["neg_selected_redrow"], name="neg_selected_redrow"
    )
    graph.node.insert(neg_index + 1, bitnot)

    hit = next(node for node in graph.node if "hit" in node.output)
    assert hit.op_type == "Greater" and list(hit.input) == ["hm", "zero_u8"]
    hit.op_type = "Cast"
    del hit.input[:]
    hit.input.extend(["hm"])
    hit.attribute.extend([onnx.helper.make_attribute("to", onnx.TensorProto.BOOL)])

    kept = [init for init in graph.initializer if init.name != "zero_u8"]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    save_checked(model, "task048_not_sub_and_bool_cast.onnx")


def task392_unsqueeze_axes_attribute() -> None:
    """Downgrade to opset 12 so Unsqueeze axes are an unscored attribute."""
    model = onnx.load(HERE / "baseline" / "task392.onnx")
    assert len(model.opset_import) == 1 and model.opset_import[0].version == 14
    model.opset_import[0].version = 12
    graph = model.graph
    node = graph.node[8]
    assert node.op_type == "Unsqueeze" and list(node.input) == ["ri", "uaxes"]
    del node.input[:]
    node.input.extend(["ri"])
    node.attribute.extend([onnx.helper.make_attribute("axes", [0, 1, 3])])
    kept = [init for init in graph.initializer if init.name != "uaxes"]
    assert len(kept) + 1 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    save_checked(model, "task392_unsqueeze_axes_attribute.onnx")


def _replace_output_chain(graph: onnx.GraphProto, old_outputs: set[str], new_node: onnx.NodeProto) -> None:
    """Replace the first node in an output chain and delete the remainder."""
    indexes = [i for i, node in enumerate(graph.node) if set(node.output) & old_outputs]
    assert indexes
    first = min(indexes)
    kept = [node for i, node in enumerate(graph.node) if i not in set(indexes)]
    kept.insert(first, new_node)
    del graph.node[:]
    graph.node.extend(kept)


def task037_binary_rule_fusion() -> None:
    """Fuse binary AND/OR chains while retaining every shape cloak."""
    model = onnx.load(HERE / "baseline" / "task037.onnx")
    graph = model.graph

    for prefix in ("pp", "mm", "pm", "mp"):
        old = {f"{prefix}_reach{i}" for i in range(1, 6)}
        node = onnx.helper.make_node(
            "Min",
            [f"{prefix}{i}_comp" for i in range(1, 6)],
            [f"{prefix}_reach5"],
            name=f"{prefix}_reach5",
        )
        _replace_output_chain(graph, old, node)

    # P <= -M for binary P,M is true iff both are zero.  Compute
    # 1 - max(P,M) directly for the main and anti-diagonal branches.
    _replace_output_chain(
        graph,
        {"main_left_false", "main_flag", "main_pos"},
        onnx.helper.make_node(
            "Max", ["pp_reach5", "mm_reach5"], ["main_union"], name="main_union"
        ),
    )
    insert_at = next(i for i, n in enumerate(graph.node) if "main_union" in n.output) + 1
    graph.node.insert(
        insert_at,
        onnx.helper.make_node(
            "HardSigmoid",
            ["main_union"],
            ["main_pos"],
            name="main_pos",
            alpha=-1.0,
            beta=1.0,
        ),
    )
    _replace_output_chain(
        graph,
        {"anti_left_false", "anti_flag", "anti_pos"},
        onnx.helper.make_node(
            "Max", ["pm_reach5", "mp_reach5"], ["anti_union"], name="anti_union"
        ),
    )
    insert_at = next(i for i, n in enumerate(graph.node) if "anti_union" in n.output) + 1
    graph.node.insert(
        insert_at,
        onnx.helper.make_node(
            "HardSigmoid",
            ["anti_union"],
            ["anti_pos"],
            name="anti_pos",
            alpha=-1.0,
            beta=1.0,
        ),
    )

    # For binary tensors, the former complement/compare/cast triples are OR.
    _replace_output_chain(
        graph,
        {"colored_main_comp", "colored_main_flag", "colored_main"},
        onnx.helper.make_node(
            "Max", ["endpoint_compact", "main_pos"], ["colored_main"], name="colored_main"
        ),
    )
    _replace_output_chain(
        graph,
        {"colored_comp", "colored_flag", "colored"},
        onnx.helper.make_node(
            "Max", ["colored_main", "anti_pos"], ["colored"], name="colored"
        ),
    )

    produced = {output for node in graph.node for output in node.output}
    value_info = [vi for vi in graph.value_info if vi.name in produced]
    # Add value-info for the two new scalar-cloaked branch unions.
    ref = next(vi for vi in graph.value_info if vi.name == "main_pos")
    for name in ("main_union", "anti_union"):
        vi = onnx.ValueInfoProto()
        vi.CopyFrom(ref)
        vi.name = name
        value_info.append(vi)
    del graph.value_info[:]
    graph.value_info.extend(value_info)
    save_checked(model, "task037_binary_rule_fusion.onnx")


def task037_bool_rule_pipeline() -> None:
    """Keep allocation topology intact but evaluate the binary rule in bool."""
    model = onnx.load(HERE / "baseline" / "task037.onnx")
    graph = model.graph
    by_output = {output: node for node in graph.node for output in node.output}

    # The axial i8 shifts are already binary.  Convert directly to bool and
    # complement with Not, replacing two f16 tensors by two bool tensors.
    for prefix in ("pp", "mm"):
        for i in range(1, 6):
            half_name = f"{prefix}{i}_half"
            cast = by_output[half_name]
            cast.op_type = "Cast"
            del cast.input[1:]
            del cast.attribute[:]
            cast.attribute.extend([onnx.helper.make_attribute("to", onnx.TensorProto.BOOL)])
            comp = by_output[f"{prefix}{i}_comp"]
            comp.op_type = "Not"
            del comp.attribute[:]

    # Diagonal shifts are f16 binary tensors; equality to the existing f16
    # zero anchor is their exact boolean complement.
    for prefix in ("pm", "mp"):
        for i in range(1, 6):
            comp = by_output[f"{prefix}{i}_comp"]
            comp.op_type = "Equal"
            del comp.attribute[:]
            comp.input.extend(["half"])

    # Preserve all five allocation steps in each reachability chain.
    for prefix in ("pp", "mm", "pm", "mp"):
        first = by_output[f"{prefix}_reach1"]
        first.op_type = "Identity"
        del first.attribute[:]
        for i in range(2, 6):
            node = by_output[f"{prefix}_reach{i}"]
            node.op_type = "And"
            del node.attribute[:]

    # With positive boolean reaches, the former comparison is !(P || M).
    for branch, left, right in (
        ("main", "pp_reach5", "mm_reach5"),
        ("anti", "pm_reach5", "mp_reach5"),
    ):
        union = by_output[f"{branch}_left_false"]
        union.op_type = "Or"
        del union.input[:]
        union.input.extend([left, right])
        del union.attribute[:]
        invert = by_output[f"{branch}_flag"]
        invert.op_type = "Not"
        del invert.input[:]
        invert.input.extend([f"{branch}_left_false"])
        del invert.attribute[:]
        cast = by_output[f"{branch}_pos"]
        cast.op_type = "Cast"
        del cast.input[:]
        cast.input.extend([f"{branch}_flag"])
        del cast.attribute[:]
        cast.attribute.extend([onnx.helper.make_attribute("to", onnx.TensorProto.FLOAT16)])

    bool_names = set()
    for prefix in ("pp", "mm"):
        bool_names.update(f"{prefix}{i}_half" for i in range(1, 6))
    for prefix in ("pp", "mm", "pm", "mp"):
        bool_names.update(f"{prefix}{i}_comp" for i in range(1, 6))
        bool_names.update(f"{prefix}_reach{i}" for i in range(1, 6))
    bool_names.update({"main_left_false", "anti_left_false"})
    for vi in graph.value_info:
        if vi.name in bool_names:
            vi.type.tensor_type.elem_type = onnx.TensorProto.BOOL

    save_checked(model, "task037_bool_rule_pipeline.onnx")


if __name__ == "__main__":
    # task037_direct_shape_vector() is intentionally not emitted: direct shape
    # inference conflicts with the baseline's downstream shape cloaks.
    task048_not_sub_and_bool_cast()
    # task392_unsqueeze_axes_attribute() cannot retain int8 Sub at opset 12.
    task037_binary_rule_fusion()
    task037_bool_rule_pipeline()
