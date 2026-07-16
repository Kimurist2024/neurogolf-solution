#!/usr/bin/env python3
"""Build exact-algebra task379 candidates from the exact 8002.63 member."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline" / "task379.onnx"
CANDIDATES = HERE / "candidates"


def build_without_terminal_ones() -> Path:
    """Remove redundant all-one E operands from the terminal Einsum.

    Each removed E has one spatial output label already supplied by B and one
    reduction label already supplied by another initializer.  Since E is all
    ones, deleting it leaves exactly the same Einstein summation.
    """
    model = onnx.load(SOURCE)
    node = model.graph.node[-1]
    assert node.op_type == "Einsum" and node.output == ["output"]
    equation = helper.get_attribute_value(next(a for a in node.attribute if a.name == "equation")).decode()
    terms, output = equation.split("->")
    term_list = terms.split(",")
    assert len(term_list) == len(node.input)
    remove_indices = [index for index, name in enumerate(node.input) if name == "E"]
    assert remove_indices == [7, 8, 19, 20, 23]
    node.input[:] = [name for index, name in enumerate(node.input) if index not in remove_indices]
    term_list = [term for index, term in enumerate(term_list) if index not in remove_indices]
    for attr in node.attribute:
        if attr.name == "equation":
            attr.s = (",".join(term_list) + "->" + output).encode()
    keep = [initializer for initializer in model.graph.initializer if initializer.name != "E"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path = CANDIDATES / "task379_drop_redundant_E.onnx"
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return output_path


def build_without_identity_mode() -> Path:
    """Remove the redundant identity mode transform used to spell QV as NV."""
    model = onnx.load(SOURCE)
    node = model.graph.node[-1]
    equation_attr = next(a for a in node.attribute if a.name == "equation")
    terms, output = equation_attr.s.decode().split("->")
    term_list = terms.split(",")
    assert len(term_list) == len(node.input)
    identity = "NV__mode1__from__QV"
    identity_indices = [index for index, name in enumerate(node.input) if name == identity]
    assert identity_indices == [3, 15]

    # QV[x,m,g] I[i,m] == QV[x,i,g]
    # QV[u,f,e] I[i,f] == QV[u,i,e]
    assert term_list[2:4] == ["xmg", "im"]
    assert term_list[14:16] == ["ufe", "if"]
    term_list[2] = "xig"
    term_list[14] = "uie"
    node.input[:] = [name for index, name in enumerate(node.input) if index not in identity_indices]
    term_list = [term for index, term in enumerate(term_list) if index not in identity_indices]
    equation_attr.s = (",".join(term_list) + "->" + output).encode()

    keep = [initializer for initializer in model.graph.initializer if initializer.name != identity]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path = CANDIDATES / "task379_drop_identity_mode.onnx"
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return output_path


def build_fold_mode_into_m2() -> Path:
    """Precontract the repeated 3x3 mode transform into the shared M2 tensor."""
    model = onnx.load(SOURCE)
    by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    m2 = numpy_helper.to_array(by_name["M2"])
    transform_name = "NV__mode1__from__QV"
    transform = numpy_helper.to_array(by_name[transform_name])
    # [o,t,i,a] * [i,m] -> [o,t,m,a].  The same M2 is used by both branches.
    folded = np.einsum("otia,im->otma", m2, transform, optimize=False).astype(m2.dtype)
    by_name["M2"].CopyFrom(numpy_helper.from_array(folded, "M2"))

    node = model.graph.node[-1]
    equation_attr = next(a for a in node.attribute if a.name == "equation")
    terms, output = equation_attr.s.decode().split("->")
    term_list = terms.split(",")
    identity_indices = [index for index, name in enumerate(node.input) if name == transform_name]
    assert identity_indices == [3, 15]
    assert term_list[0] == "otia" and term_list[2:4] == ["xmg", "im"]
    assert term_list[12] == "ptid" and term_list[14:16] == ["ufe", "if"]
    term_list[0] = "otma"
    term_list[12] = "ptfd"
    node.input[:] = [name for index, name in enumerate(node.input) if index not in identity_indices]
    term_list = [term for index, term in enumerate(term_list) if index not in identity_indices]
    equation_attr.s = (",".join(term_list) + "->" + output).encode()

    keep = [initializer for initializer in model.graph.initializer if initializer.name != transform_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path = CANDIDATES / "task379_fold_mode_into_M2.onnx"
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return output_path


def build_threshold_orientation() -> Path:
    """Use the exact reachable-domain threshold for the orientation split.

    The task generator produces ``diff2 == 14`` or ``24`` for vertical cases
    and ``diff2 > 24`` for horizontal cases.  Replacing the two Equal tests and
    their Or with a Greater/Not pair preserves the downstream tensor names and
    removes one scalar parameter plus one scalar intermediate.
    """
    model = onnx.load(SOURCE)
    rebuilt = []
    for node in model.graph.node:
        outputs = set(node.output)
        if outputs & {"eq_small", "nhb", "is_h"}:
            if "is_h" in outputs:
                rebuilt.append(
                    helper.make_node(
                        "Greater",
                        ["diff2", "d2_big"],
                        ["is_h"],
                        name="orientation_is_horizontal",
                    )
                )
                rebuilt.append(
                    helper.make_node(
                        "Not",
                        ["is_h"],
                        ["nhb"],
                        name="orientation_not_horizontal",
                    )
                )
            continue
        rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)

    keep = [initializer for initializer in model.graph.initializer if initializer.name != "d2_small"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path = CANDIDATES / "task379_threshold_orientation.onnx"
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return output_path


def build_affine_orientation() -> Path:
    """Eliminate the explicit boolean complement via an affine O basis.

    Every Where using ``is_h`` can instead use ``nhb`` with its branches
    swapped.  The terminal orientation vector changes from ``[1-nhb, nhb]``
    to ``[1, nhb]``; the first mode of M2 is retained and the second becomes
    ``old_second - old_first`` so the mathematical contraction is unchanged.
    This candidate is intentionally subjected to raw-output comparison because
    FP16 Einsum reassociation can still make an algebraic identity unsafe.
    """
    model = onnx.load(SOURCE)
    by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    m2 = numpy_helper.to_array(by_name["M2"])
    transformed = m2.copy()
    transformed[1] = (m2[1] - m2[0]).astype(m2.dtype)
    by_name["M2"].CopyFrom(numpy_helper.from_array(transformed, "M2"))

    rebuilt = []
    for node in model.graph.node:
        if node.output == ["is_h"]:
            assert node.op_type == "Not" and node.input == ["nhb"]
            continue
        if node.op_type == "Where" and node.input[0] == "is_h":
            node.input[0] = "nhb"
            node.input[1], node.input[2] = node.input[2], node.input[1]
        if node.output == ["Ob"]:
            assert node.op_type == "Concat" and list(node.input) == ["is_h", "nhb"]
            rebuilt.append(helper.make_node("Cast", ["nhb"], ["nh16"], to=10, name="nh16"))
            continue
        if node.output == ["O"]:
            assert node.op_type == "Cast" and node.input == ["Ob"]
            rebuilt.append(
                helper.make_node(
                    "Concat", ["oneh", "nh16"], ["O"], axis=0, name="orientation_affine"
                )
            )
            continue
        rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path = CANDIDATES / "task379_affine_orientation.onnx"
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return output_path


def build_qv_middle_rank2() -> Path:
    """Factor QV's duplicated middle row through two exact one-hot maps.

    QV's middle rows are ``[q0, q1, q0]``.  The existing NV map instead uses
    ``[q0, q1, q1]``.  A two-row Q core plus the two 3x2 maps and a two-element
    row selector costs 22 parameters versus the incumbent QV/NV/Rflip cost 24.
    The added one-hot contractions are exact algebraically, but raw equivalence
    remains mandatory because the terminal Einsum is evaluated in FP16.
    """
    model = onnx.load(SOURCE)
    by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    qv = numpy_helper.to_array(by_name["QV"])
    qcore = qv[:, :2, :].copy()
    expand = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=qv.dtype)
    mode = np.asarray([[1, 0], [0, 1], [0, 1]], dtype=qv.dtype)
    row1 = np.asarray([0, 1], dtype=qv.dtype)
    remove = {"QV", "NV__mode1__from__QV", "Rflip__slice1_1__of__QV"}
    keep = [initializer for initializer in model.graph.initializer if initializer.name not in remove]
    keep.extend(
        [
            numpy_helper.from_array(qcore, "QCore2"),
            numpy_helper.from_array(expand, "QExpand3x2"),
            numpy_helper.from_array(mode, "QMode3x2"),
            numpy_helper.from_array(row1, "QRow1_2"),
        ]
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)

    terminal = model.graph.node[-1]
    equation_attr = next(attr for attr in terminal.attribute if attr.name == "equation")
    terms, output = equation_attr.s.decode().split("->")
    old_terms = terms.split(",")
    assert len(old_terms) == len(terminal.input)
    new_inputs: list[str] = []
    new_terms: list[str] = []
    for index, (name, term) in enumerate(zip(terminal.input, old_terms)):
        if name == "QV":
            name = "QCore2"
        elif name == "NV__mode1__from__QV":
            name = "QMode3x2"
        elif name == "Rflip__slice1_1__of__QV":
            name = "QRow1_2"
        if index == 5:
            assert term == "yih"
            term = "yqh"
        elif index == 17:
            assert term == "vij"
            term = "vzj"
        new_inputs.append(name)
        new_terms.append(term)
        if index == 5:
            new_inputs.append("QExpand3x2")
            new_terms.append("iq")
        elif index == 17:
            new_inputs.append("QExpand3x2")
            new_terms.append("iz")
    terminal.input[:] = new_inputs
    equation_attr.s = (",".join(new_terms) + "->" + output).encode()
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path = CANDIDATES / "task379_qv_middle_rank2.onnx"
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return output_path


def main() -> None:
    print(build_without_terminal_ones())
    print(build_without_identity_mode())
    print(build_fold_mode_into_m2())
    print(build_threshold_orientation())
    print(build_affine_orientation())
    print(build_qv_middle_rank2())


if __name__ == "__main__":
    main()
