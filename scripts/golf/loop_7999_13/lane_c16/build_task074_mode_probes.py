#!/usr/bin/env python3
"""Build the feasible two-label versions of the task074 mode-reuse idea.

The exact output-direct Einsum already consumes 50 of the 52 legal index
letters.  Replacing all eight uses of one feature initializer with a mode
transform needs eight independent contraction labels.  These probes reuse the
two remaining labels in several natural partitions, making the resulting
semantic coupling explicit and testable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
BASE = HERE / "base" / "task074.onnx"
OUT = HERE / "task074_mode_probes"


def equation_attr(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attr for attr in node.attribute if attr.name == "equation")


def transform(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    target_mode = target[0].astype(np.float64)
    source_mode = source[0].astype(np.float64)
    matrix = target_mode @ np.linalg.pinv(source_mode)
    rounded = np.rint(matrix).astype(np.float32)
    if not np.array_equal(rounded @ source_mode, target_mode):
        raise RuntimeError("mode transform is not exactly integral")
    return rounded


def rewrite(
    model: onnx.ModelProto,
    target_name: str,
    source_name: str,
    assignment: str,
) -> dict[str, object]:
    node = model.graph.node[0]
    attr = equation_attr(node)
    lhs, rhs = attr.s.decode("ascii").split("->", 1)
    operands = lhs.split(",")
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    mode = transform(arrays[target_name], arrays[source_name])
    mode_name = f"{target_name}_from_{source_name}_mode"
    model.graph.initializer.append(numpy_helper.from_array(mode, mode_name))

    positions = [index for index, name in enumerate(node.input) if name == target_name]
    if len(positions) != len(assignment):
        raise RuntimeError(f"expected {len(assignment)} uses, found {len(positions)}")
    inputs = list(node.input)
    for use_index, position in reversed(list(enumerate(positions))):
        original = operands[position]
        latent = assignment[use_index]
        source_subscript = original[0] + latent + original[2]
        transform_subscript = original[1] + latent
        inputs[position : position + 1] = [source_name, mode_name]
        operands[position : position + 1] = [source_subscript, transform_subscript]
    del node.input[:]
    node.input.extend(inputs)
    attr.s = (",".join(operands) + "->" + rhs).encode("ascii")
    kept = [item for item in model.graph.initializer if item.name != target_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return {
        "target": target_name,
        "source": source_name,
        "uses": len(positions),
        "assignment": assignment,
        "mode": mode.tolist(),
        "target_params": int(arrays[target_name].size),
        "transform_params": int(mode.size),
        "nominal_saving": int(arrays[target_name].size - mode.size),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    patterns = {
        "all_y": "YYYYYYYY",
        "halves": "YYYYZZZZ",
        "alternating": "YZYZYZYZ",
        "pairs": "YYZZYYZZ",
        "cross_pairs": "YZZYYZZY",
    }
    rows: list[dict[str, object]] = []
    for target, source in (("Tfeat", "Bfeat"), ("Bfeat", "Tfeat")):
        for label, assignment in patterns.items():
            model = onnx.load(BASE)
            record: dict[str, object] = {
                "variant": f"{target.lower()}_from_{source.lower()}_{label}"
            }
            try:
                record.update(rewrite(model, target, source, assignment))
                onnx.checker.check_model(model, full_check=True)
                onnx.shape_inference.infer_shapes(
                    model, strict_mode=True, data_prop=True
                )
                path = OUT / f"{record['variant']}.onnx"
                onnx.save(model, path)
                record.update(
                    built=True,
                    path=str(path.relative_to(HERE.parents[3])),
                    operands=len(model.graph.node[0].input),
                )
            except Exception as exc:  # noqa: BLE001
                record.update(built=False, error=f"{type(exc).__name__}: {exc}")
            rows.append(record)
    (HERE / "task074_mode_probe_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
