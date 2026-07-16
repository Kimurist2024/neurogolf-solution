#!/usr/bin/env python3
"""Build bounded A32 probes from the pinned Wave16 members."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
BASE_ZIP = HERE.parent / "submission_7999.13_wave16_candidate_meta.zip"
EXPECTED_SHA = "4014cbafea4862f67ebf5ff24be13149b45b333c95bfa680be7216f001a6bb3a"


def extract(task: int) -> onnx.ModelProto:
    assert hashlib.sha256(BASE_ZIP.read_bytes()).hexdigest() == EXPECTED_SHA
    with zipfile.ZipFile(BASE_ZIP) as archive:
        payload = archive.read(f"task{task:03d}.onnx")
    (HERE / f"task{task:03d}_base.onnx").write_bytes(payload)
    return onnx.load_model_from_string(payload)


def equation(node: onnx.NodeProto) -> tuple[list[str], str, onnx.AttributeProto]:
    attribute = next(item for item in node.attribute if item.name == "equation")
    lhs, rhs = attribute.s.decode("ascii").split("->")
    return lhs.split(","), rhs, attribute


def save(model: onnx.ModelProto, name: str) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, HERE / name)


def couple_selected_states(base: onnx.ModelProto, *, eliminate_s: bool) -> onnx.ModelProto:
    """Couple three one-hot S contractions; optionally derive S as a C row sum."""
    model = onnx.ModelProto()
    model.CopyFrom(base)
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
        elif index == 12 and eliminate_s:
            name, term = "C", "ND"  # D was freed by coupling; sum over all colors.
        rebuilt.append((name, term))

    del node.input[:]
    node.input.extend(name for name, _ in rebuilt)
    attribute.s = (",".join(term for _, term in rebuilt) + "->" + rhs).encode("ascii")

    if eliminate_s:
        for index, item in enumerate(model.graph.initializer):
            if item.name == "C":
                c = numpy_helper.to_array(item).copy()
                # Move the unused color-1/3 half-features from latent row 2 to
                # row 1.  Columns 0,2,4,8 (the complete generator palette)
                # remain bit-identical, while row_sum(C) becomes [0,0,-1]=S.
                c[1, 1] = c[2, 1]
                c[1, 3] = c[2, 3]
                c[2, 1] = 0.0
                c[2, 3] = 0.0
                model.graph.initializer[index].CopyFrom(numpy_helper.from_array(c, "C"))
                assert np.array_equal(c.sum(axis=1), np.array([0.0, 0.0, -1.0], np.float32))
                break
        kept = [item for item in model.graph.initializer if item.name != "S"]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
    return model


def drop_s_and_one_selected_c(base: onnx.ModelProto, removed_term: str) -> onnx.ModelProto:
    """Probe whether one of the three coupled color tests is generator-redundant."""
    model = couple_selected_states(base, eliminate_s=False)
    node = model.graph.node[0]
    terms, rhs, attribute = equation(node)
    pairs = list(zip(node.input, terms))
    assert removed_term in {"NK", "NA", "NW"}
    rebuilt = [
        (name, term)
        for name, term in pairs
        if not (name == "S" and term == "N") and not (name == "C" and term == removed_term)
    ]
    assert len(pairs) - len(rebuilt) == 2
    del node.input[:]
    node.input.extend(name for name, _ in rebuilt)
    attribute.s = (",".join(term for _, term in rebuilt) + "->" + rhs).encode("ascii")
    kept = [item for item in model.graph.initializer if item.name != "S"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model


def share_small_tensor(base: onnx.ModelProto, retained: str) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.CopyFrom(base)
    removed = "M01" if retained == "B" else "B"
    node = model.graph.node[0]
    for index, name in enumerate(node.input):
        if name == removed:
            node.input[index] = retained
    kept = [item for item in model.graph.initializer if item.name != removed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model


def scalarize_coupled_s(base: onnx.ModelProto) -> onnx.ModelProto:
    """After coupling, test whether C's sparse palette already fixes state N=2."""
    model = couple_selected_states(base, eliminate_s=False)
    node = model.graph.node[0]
    terms, rhs, attribute = equation(node)
    for index, (name, term) in enumerate(zip(node.input, terms)):
        if name == "S":
            assert term == "N"
            terms[index] = "D"  # Free, private size-one bond: scalar -1.
    for index, item in enumerate(model.graph.initializer):
        if item.name == "S":
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(np.array([-1.0], dtype=np.float32), "S")
            )
            break
    attribute.s = (",".join(terms) + "->" + rhs).encode("ascii")
    return model


def main() -> None:
    extract(288)
    task335 = extract(335)
    save(couple_selected_states(task335, eliminate_s=False), "task335_coupled_s.onnx")
    save(couple_selected_states(task335, eliminate_s=True), "task335_coupled_c_rowsum.onnx")
    for term in ("NK", "NA", "NW"):
        save(
            drop_s_and_one_selected_c(task335, term),
            f"task335_drop_s_and_{term.lower()}.onnx",
        )
    save(share_small_tensor(task335, "B"), "task335_share_b.onnx")
    save(share_small_tensor(task335, "M01"), "task335_share_m01.onnx")
    save(scalarize_coupled_s(task335), "task335_coupled_scalar_s.onnx")


if __name__ == "__main__":
    main()
