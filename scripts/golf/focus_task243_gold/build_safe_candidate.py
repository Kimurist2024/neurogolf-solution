#!/usr/bin/env python3
"""Build the fully gated truthful task243 candidate.

This keeps the exact fixed relation matrices, splits the giant contraction to
gain 16 additional flood steps, strengthens the blue seed, normalizes every
relation factor to avoid float32 overflow, and applies Sign at the output so
the public/private threshold margin is exactly one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "candidates/task243_truthful_chunked_plus16.onnx"
OUTPUT = HERE / "candidates/task243_truthful_safe.onnx"
EVIDENCE = HERE / "safe_build.json"
BLUE_WEIGHT = -2048.0
RELATION_SCALE = 0.875  # exact binary 7/8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_initializer(
    model: onnx.ModelProto, name: str, array: np.ndarray
) -> None:
    index = next(
        index for index, init in enumerate(model.graph.initializer) if init.name == name
    )
    model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array, name))


def main() -> None:
    model = onnx.load(SOURCE)
    weights_init = next(init for init in model.graph.initializer if init.name == "w_dyn")
    weights = numpy_helper.to_array(weights_init).copy()
    weights[1] = np.float32(BLUE_WEIGHT)
    replace_initializer(model, "w_dyn", weights)

    relation_init = next(init for init in model.graph.initializer if init.name == "L")
    relation = numpy_helper.to_array(relation_init).copy()
    relation *= np.float32(RELATION_SCALE)
    replace_initializer(model, "L", relation)

    terminal = model.graph.node[-1]
    if terminal.op_type != "Einsum" or list(terminal.output) != ["output"]:
        raise RuntimeError("unexpected chunked terminal")
    terminal.output[0] = "raw_output"
    terminal.name = "truthful_walk_chunk1_plus16_normalized"
    model.graph.node.append(
        helper.make_node("Sign", ["raw_output"], ["output"], name="stable_sign")
    )
    del model.graph.value_info[:]
    model.graph.value_info.extend(
        [
            helper.make_tensor_value_info(
                "reach", TensorProto.FLOAT, [1, 10, 30, 30]
            ),
            helper.make_tensor_value_info(
                "raw_output", TensorProto.FLOAT, [1, 10, 30, 30]
            ),
        ]
    )
    model.producer_name = "codex-task243-truthful-safe-repair"

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    shapes = {
        value.name: [
            int(dim.dim_value) for dim in value.type.tensor_type.shape.dim
        ]
        for value in [
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        ]
    }
    expected = [1, 10, 30, 30]
    for name in ("input", "reach", "raw_output", "output"):
        if shapes.get(name) != expected:
            raise RuntimeError(f"non-truthful shape for {name}: {shapes.get(name)}")

    onnx.save(model, OUTPUT)
    EVIDENCE.write_text(
        json.dumps(
            {
                "task": 243,
                "source": str(SOURCE.relative_to(ROOT)),
                "source_sha256": sha256(SOURCE),
                "candidate": str(OUTPUT.relative_to(ROOT)),
                "candidate_sha256": sha256(OUTPUT),
                "blue_weight": BLUE_WEIGHT,
                "relation_scale": RELATION_SCALE,
                "relation_scale_exact_binary": True,
                "additional_flood_steps": 16,
                "output_stabilizer": "Sign",
                "truthful_shapes": shapes,
                "full_check": True,
                "strict_shape_inference_data_prop": True,
                "nodes": len(model.graph.node),
                "params_by_shape": 1010,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
