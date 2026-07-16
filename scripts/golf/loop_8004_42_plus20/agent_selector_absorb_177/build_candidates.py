#!/usr/bin/env python3
"""Build exact selector/precontraction controls for tasks 246, 335, and 348.

The controls deliberately do not touch the root submission.  They make the
local tensor-network alternatives concrete so that the audit can compare
actual parameter counts instead of relying on informal estimates.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline"
CONTROLS = HERE / "audit_controls"


def equation(node: onnx.NodeProto) -> tuple[list[str], str, onnx.AttributeProto]:
    attribute = next(item for item in node.attribute if item.name == "equation")
    lhs, rhs = attribute.s.decode("ascii").split("->")
    return lhs.split(","), rhs, attribute


def set_equation(
    node: onnx.NodeProto,
    terms: list[str],
    rhs: str,
    attribute: onnx.AttributeProto,
) -> None:
    attribute.s = (",".join(terms) + "->" + rhs).encode("ascii")


def replace_initializers(
    model: onnx.ModelProto,
    removed: set[str],
    added: list[onnx.TensorProto],
) -> None:
    kept = [item for item in model.graph.initializer if item.name not in removed]
    del model.graph.initializer[:]
    model.graph.initializer.extend([*kept, *added])


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def couple_selector(base: onnx.ModelProto) -> onnx.ModelProto:
    """Couple the three independent S=-e2 contractions into one exact bond.

    Original:
      sum_N C[N,K]S[N] * sum_D C[D,A]S[D] * sum_x C[x,W]S[x]
    Since S has the sole nonzero S[2]=-1, this equals
      sum_N C[N,K]C[N,A]C[N,W]S[N].
    The rewrite changes operand count only; the unique initializer set is the
    same, so the scorer parameter count cannot fall.
    """
    model = copy.deepcopy(base)
    node = model.graph.node[0]
    terms, rhs, attribute = equation(node)
    pairs = list(zip(node.input, terms))
    assert pairs[11:13] == [("C", "NK"), ("S", "N")]
    assert pairs[23:25] == [("C", "DA"), ("S", "D")]
    assert pairs[37:39] == [("C", "xW"), ("S", "x")]

    rebuilt: list[tuple[str, str]] = []
    for index, (name, term) in enumerate(pairs):
        if index in (24, 38):
            continue
        if index == 23:
            term = "NA"
        elif index == 37:
            term = "NW"
        rebuilt.append((name, term))
    del node.input[:]
    node.input.extend(name for name, _ in rebuilt)
    set_equation(node, [term for _, term in rebuilt], rhs, attribute)
    model.graph.name += "_selector_coupled"
    return model


def absorb_selector(base: onnx.ModelProto) -> onnx.ModelProto:
    """Precompute T[k]=sum_n S[n]C[n,k] for all three S/C pairs."""
    model = copy.deepcopy(base)
    values = arrays(model)
    t = np.einsum("n,nk->k", values["S"], values["C"], optimize=False)
    node = model.graph.node[0]
    terms, rhs, attribute = equation(node)
    pairs = list(zip(node.input, terms))
    replacements = {11: "K", 23: "A", 37: "W"}
    skipped = {12, 24, 38}
    rebuilt: list[tuple[str, str]] = []
    for index, pair in enumerate(pairs):
        if index in skipped:
            continue
        if index in replacements:
            rebuilt.append(("T", replacements[index]))
        else:
            rebuilt.append(pair)
    del node.input[:]
    node.input.extend(name for name, _ in rebuilt)
    set_equation(node, [term for _, term in rebuilt], rhs, attribute)
    replace_initializers(model, {"S"}, [numpy_helper.from_array(t.astype(np.float32), "T")])
    model.graph.name += "_selector_absorbed"
    return model


def precompose_duplicate_c(base: onnx.ModelProto) -> onnx.ModelProto:
    """Replace the repeated elementwise C[t,i]C[t,i] by Csq[t,i]."""
    model = copy.deepcopy(base)
    values = arrays(model)
    csq = np.square(values["C"], dtype=np.float32)
    node = model.graph.node[0]
    terms, rhs, attribute = equation(node)
    pairs = list(zip(node.input, terms))
    matches = [index for index, pair in enumerate(pairs) if pair == ("C", "ti")]
    assert matches == [41, 42]
    rebuilt: list[tuple[str, str]] = []
    for index, pair in enumerate(pairs):
        if index == matches[0]:
            rebuilt.append(("Csq", "ti"))
        elif index == matches[1]:
            continue
        else:
            rebuilt.append(pair)
    del node.input[:]
    node.input.extend(name for name, _ in rebuilt)
    set_equation(node, [term for _, term in rebuilt], rhs, attribute)
    model.graph.initializer.append(numpy_helper.from_array(csq, "Csq"))
    model.graph.name += "_duplicate_c_precomposed"
    return model


def precompose_task348_cd(base: onnx.ModelProto) -> onnx.ModelProto:
    """Precompute the two selector products C1@D and C2@D.

    Every C1/C2 occurrence is immediately contracted with D over a private
    size-three bond.  The two dense products are exact, but occupy 120 elements
    versus D+C1+C2's 102 elements.
    """
    model = copy.deepcopy(base)
    values = arrays(model)
    cd1 = values["C1"] @ values["D"]
    cd2 = values["C2"] @ values["D"]
    node = model.graph.node[0]
    terms, rhs, attribute = equation(node)
    pairs = list(zip(node.input, terms))
    rebuilt: list[tuple[str, str]] = []
    index = 0
    pair_count = 0
    while index < len(pairs):
        name, term = pairs[index]
        if name in {"C1", "C2"}:
            next_name, next_term = pairs[index + 1]
            assert next_name == "D"
            assert len(term) == len(next_term) == 2 and term[1] == next_term[0]
            rebuilt.append(("CD1" if name == "C1" else "CD2", term[0] + next_term[1]))
            pair_count += 1
            index += 2
            continue
        rebuilt.append((name, term))
        index += 1
    assert pair_count == 14
    del node.input[:]
    node.input.extend(name for name, _ in rebuilt)
    set_equation(node, [term for _, term in rebuilt], rhs, attribute)
    replace_initializers(
        model,
        {"C1", "C2", "D"},
        [
            numpy_helper.from_array(cd1.astype(np.float32), "CD1"),
            numpy_helper.from_array(cd2.astype(np.float32), "CD2"),
        ],
    )
    model.graph.name += "_cd_precomposed"
    return model


def save(model: onnx.ModelProto, name: str) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, CONTROLS / name)


def main() -> None:
    CONTROLS.mkdir(parents=True, exist_ok=True)
    for task in (246, 335):
        base = onnx.load(BASELINE / f"task{task:03d}.onnx")
        save(couple_selector(base), f"task{task:03d}_coupled_equal109.onnx")
        save(absorb_selector(base), f"task{task:03d}_absorbed_higher116.onnx")
        save(precompose_duplicate_c(base), f"task{task:03d}_csquare_higher139.onnx")
    base348 = onnx.load(BASELINE / "task348.onnx")
    save(precompose_task348_cd(base348), "task348_cd_precomposed_higher148.onnx")


if __name__ == "__main__":
    main()
