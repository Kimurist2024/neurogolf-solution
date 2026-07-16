#!/usr/bin/env python3
"""Build exact graph-only parameter shaves for structural screening.

These files stay in this lane.  They are not merge artifacts: both source
graphs use false intermediate shape declarations, so the numeric shaves must
still pass the truthful-runtime-shape gate (and are expected to fail it).
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


HERE = Path(__file__).resolve().parent


def replace_castlike_with_cast(model: onnx.ModelProto, target: str, elem_type: int) -> int:
    count = 0
    for node in model.graph.node:
        if node.op_type == "CastLike" and len(node.input) == 2 and node.input[1] == target:
            del node.input[1:]
            node.op_type = "Cast"
            node.domain = ""
            node.attribute.extend([helper.make_attribute("to", elem_type)])
            count += 1
    return count


def build(task: int, target: str) -> None:
    source = HERE / "base" / f"task{task:03d}.onnx"
    model = onnx.load(source)
    target_init = next(item for item in model.graph.initializer if item.name == target)
    count = replace_castlike_with_cast(model, target, target_init.data_type)
    if not count:
        raise RuntimeError(f"task{task:03d}: no CastLike target {target}")
    kept = [item for item in model.graph.initializer if item.name != target]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.name += "_cast_exact_shave"
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, HERE / "candidates" / f"task{task:03d}_cast_exact_shave.onnx")


def main() -> None:
    build(282, "bref")
    build(283, "b0")


if __name__ == "__main__":
    main()
