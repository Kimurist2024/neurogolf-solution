#!/usr/bin/env python3
"""Replace an Einsum-only factor by an exact product of stored factors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]


def equation_attribute(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attribute for attribute in node.attribute if attribute.name == "equation")


def rewrite(
    source_model: onnx.ModelProto,
    target_name: str,
    left_name: str,
    right_name: str,
) -> tuple[onnx.ModelProto, dict[str, object]]:
    model = copy.deepcopy(source_model)
    arrays = {
        tensor.name: np.asarray(numpy_helper.to_array(tensor))
        for tensor in model.graph.initializer
    }
    target = arrays[target_name]
    left = arrays[left_name]
    right = arrays[right_name]
    if target.shape != left.shape or target.shape != right.shape:
        raise ValueError("all factors must have identical shapes")
    if target.dtype != left.dtype or target.dtype != right.dtype:
        raise ValueError("all factors must have identical dtypes")
    product = np.multiply(left, right, dtype=target.dtype)
    if not np.array_equal(product, target):
        raise ValueError("stored factors do not reconstruct target exactly")

    uses = 0
    for node in model.graph.node:
        positions = [position for position, name in enumerate(node.input) if name == target_name]
        if not positions:
            continue
        if node.op_type != "Einsum":
            raise ValueError(f"non-Einsum use of {target_name}: {node.op_type}")
        attribute = equation_attribute(node)
        text = attribute.s.decode("ascii")
        lhs, rhs = text.split("->", 1)
        terms = lhs.split(",")
        inputs = list(node.input)
        for position in reversed(positions):
            term = terms[position]
            terms[position : position + 1] = [term, term]
            inputs[position : position + 1] = [left_name, right_name]
            uses += 1
        del node.input[:]
        node.input.extend(inputs)
        attribute.s = (",".join(terms) + "->" + rhs).encode("ascii")
    if uses == 0:
        raise ValueError(f"no uses of {target_name}")
    remaining = Counter(name for node in model.graph.node for name in node.input if name)
    if remaining[target_name]:
        raise ValueError(f"failed to replace all uses of {target_name}")
    kept = [tensor for tensor in model.graph.initializer if tensor.name != target_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model, {
        "target": target_name,
        "left": left_name,
        "right": right_name,
        "uses": uses,
        "removed_parameters": int(target.size),
        "added_parameters": 0,
        "net_parameter_saving": int(target.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(baseline) as archive:
        task = 158
        source = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
        model, change = rewrite(source, "coord2", "coord", "coord")
        path = out_dir / f"task{task:03d}.onnx"
        onnx.save(model, path)
        rows.append(
            {
                "task": task,
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "change": change,
            }
        )
    document = {"baseline": str(baseline.relative_to(ROOT)), "rows": rows}
    (out_dir / "build_manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
