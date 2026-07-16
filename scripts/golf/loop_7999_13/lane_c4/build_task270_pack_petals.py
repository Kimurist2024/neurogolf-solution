#!/usr/bin/env python3
"""Pack task270's two petal-colour moment streams into one Einsum lane.

The generator has petal colours 3 and 7.  Every per-flower moment consumed by
the incumbent is an integer: count <= 4, first moments <= 56, and squared-row
moment <= 784.  Encoding ``m3 + 2048*m7`` is exactly representable in float32.
Casting the packed scalar to uint8 recovers ``m3 mod 256``; quantizing with a
2048 scale recovers m7 because m3/2048 < 1/2.  This preserves the incumbent's
intentional uint8 moment arithmetic while sharing its 10-element selector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "base/task270.onnx"
OUTPUT = HERE / "task270_pack_petals.onnx"


def main() -> None:
    model = onnx.load(SOURCE)

    psel = next(init for init in model.graph.initializer if init.name == "Psel")
    packed = np.zeros((1, 10), dtype=np.float32)
    packed[0, 3] = 1.0
    packed[0, 7] = 2048.0
    psel.CopyFrom(numpy_helper.from_array(packed, name="Psel"))
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(2048.0, dtype=np.float32), name="petal_pack_scale")
    )

    # The four moment nodes retain equation ``...tc->tn`` and now emit [1,1].
    moment_outputs = {"pn", "pr", "pc", "pr2"}
    cast_nodes: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        if node.op_type == "Cast" and node.input and node.input[0] in moment_outputs:
            cast_nodes[node.input[0]] = node
    assert set(cast_nodes) == moment_outputs

    rebuilt: list[onnx.NodeProto] = []
    for node in model.graph.node:
        moment = node.input[0] if node in cast_nodes.values() else None
        if moment is None:
            rebuilt.append(node)
            continue
        original_output = node.output[0]
        low_output = f"{moment}_low8"
        node.output[0] = low_output
        rebuilt.append(node)
        high_output = f"{moment}_high8"
        if moment == "pr2":
            high_float = f"{moment}_highf"
            rebuilt.append(
                helper.make_node(
                    "Div", [moment, "petal_pack_scale"], [high_float],
                    name=f"{moment}_unpack_high_float",
                )
            )
            # pr2 can exceed 255; e.g. 290 must wrap to 34 exactly like the
            # incumbent Cast, rather than QuantizeLinear-saturating to 255.
            rebuilt.append(
                helper.make_node(
                    "Cast", [high_float], [high_output],
                    name=f"{moment}_unpack_high", to=TensorProto.UINT8,
                )
            )
        else:
            # Count and first moments are <= 56.  Their low lane is also <=56,
            # so packed/2048 is within 0.0274 of the high integer: nearest
            # quantization recovers it exactly with no saturation.
            rebuilt.append(
                helper.make_node(
                    "QuantizeLinear", [moment, "petal_pack_scale"], [high_output],
                    name=f"{moment}_unpack_high",
                )
            )
        rebuilt.append(
            helper.make_node(
                "Concat", [low_output, high_output], [original_output],
                name=f"{moment}_unpack_pair", axis=0,
            )
        )

    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    model.producer_name = "task270-packed-petal-moments"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    onnx.save(inferred, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
