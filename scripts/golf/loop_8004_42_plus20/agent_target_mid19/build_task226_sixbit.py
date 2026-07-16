#!/usr/bin/env python3
"""Rebuild task226's column decoder from the generator-minimal six probes.

The generator admits 17 width sequences.  Separator/background bits at
columns 1,2,3,5,6,8 uniquely determine all of them; columns 4 and 7 are
therefore reconstructed by exact Boolean identities over that finite, complete
generator domain.  This removes the two corresponding GatherElements indices
without changing the terminal truthful-shape renderer.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
BASE = HERE / "baseline" / "task226.onnx"
OUT = HERE / "task226_sixbit.onnx"


def main() -> None:
    model = onnx.load(BASE)

    # Remove probes c4/c7 and the old column decision DAG.  Row nodes 0..40 and
    # the truthful [1,2,10,10] renderer nodes 58..59 remain byte-semantically
    # unchanged.
    kept = []
    for node in model.graph.node:
        outputs = set(node.output)
        if outputs & {"c4_f", "c4", "c7_f", "c7"}:
            continue
        if outputs & {
            "cp1", "c4_mb", "cp2_i", "cp2_j", "cp2", "cp3_j", "cp3",
            "cp4", "cp5", "c8_lb", "cp6_i", "cp6_j", "cp6", "cp7_j",
            "cp7", "cp8", "col_code",
        }:
            continue
        kept.append(node)

    insert_at = next(i for i, node in enumerate(kept) if node.output[0] == "feat")

    col_nodes = []

    def where(cond: str, yes: str, no: str, out: str) -> str:
        col_nodes.append(helper.make_node("Where", [cond, yes, no], [out]))
        return out

    # Direct reduced ordered decision diagrams for the incumbent column-code
    # semantics.  The gathered cN values mean "background", not separator.
    # c8_mid_bg is shared by three branches, making the total exactly 25 scalar
    # Where outputs: nine bytes replace the removed two Gather+Cast pairs, while
    # idx4/idx7 remove two parameters.  Thus measured cost falls by three.
    c8_mid_bg = where("c8", "C_M", "R_B", "c8_mid_bg")
    cp1 = where("c1", "C_F", "q_xz", "cp1")

    p2_c2 = where("c2", "C_F", "q_xz", "p2_c2")
    p2_c3 = where("c3", c8_mid_bg, "R_B", "p2_c3")
    cp2 = where("c1", p2_c2, p2_c3, "cp2")

    p3_c1 = where("c1", "C_F", c8_mid_bg, "p3_c1")
    p3_c3 = where("c3", p3_c1, "q_xz", "p3_c3")
    cp3 = where("c2", p3_c3, c8_mid_bg, "cp3")

    p4_c2 = where("c2", "q_xz", "C_M", "p4_c2")
    p4_c1 = where("c1", p4_c2, "C_M", "p4_c1")
    p4_c8 = where("c8", p4_c1, "q_xz", "p4_c8")
    cp4 = where("c3", p4_c8, "C_M", "cp4")

    cp5 = where("c5", "C_M", "q_xz", "cp5")

    p6_c6 = where("c6", "C_M", "q_xz", "p6_c6")
    p6_c3 = where("c3", "C_L", "R_B", "p6_c3")
    p6_c1 = where("c1", "C_L", p6_c3, "p6_c1")
    cp6 = where("c5", p6_c6, p6_c1, "cp6")

    p7_c8_zero_mid = where("c8", "q_xz", "C_M", "p7_c8_zero_mid")
    p7_c8_last_bg = where("c8", "C_L", "R_B", "p7_c8_last_bg")
    p7_c6 = where("c6", p7_c8_zero_mid, p7_c8_last_bg, "p7_c6")
    p7_c5 = where("c5", p7_c6, "C_L", "p7_c5")
    p7_c3 = where("c3", "C_L", "q_xz", "p7_c3")
    p7_c8_right = where("c8", p7_c3, "R_B", "p7_c8_right")
    cp7 = where("c1", p7_c5, p7_c8_right, "cp7")
    cp8 = where("c8", "C_L", "q_xz", "cp8")
    col_nodes.append(
        helper.make_node(
            "Concat",
            ["C_F", cp1, cp2, cp3, cp4, cp5, cp6, cp7, cp8, "C_L"],
            ["col_code"],
            axis=3,
        )
    )

    del model.graph.node[:]
    model.graph.node.extend(kept[:insert_at] + col_nodes + kept[insert_at:])
    retained_initializers = [
        tensor for tensor in model.graph.initializer if tensor.name not in {"idx4", "idx7"}
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
