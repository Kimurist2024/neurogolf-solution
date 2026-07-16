#!/usr/bin/env python3
"""Represent constant Reduce axes as opset-17 attributes to remove parameters."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


REDUCE_OPS = {
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax",
    "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare",
}


def build(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    main_import = next(item for item in model.opset_import if not item.domain)
    original_opset = int(main_import.version)
    if original_opset < 18:
        raise RuntimeError("source opset is below 18")
    initializers = {item.name: item for item in model.graph.initializer}
    converted: list[dict[str, object]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in REDUCE_OPS or len(node.input) < 2 or not node.input[1]:
            continue
        initializer = initializers.get(node.input[1])
        if initializer is None:
            continue
        axes = np.asarray(numpy_helper.to_array(initializer), dtype=np.int64).reshape(-1).tolist()
        axes_name = node.input[1]
        del node.input[1:]
        kept_attrs = [attr for attr in node.attribute if attr.name not in {"axes", "noop_with_empty_axes"}]
        del node.attribute[:]
        node.attribute.extend(kept_attrs)
        node.attribute.append(helper.make_attribute("axes", axes))
        converted.append({"node_index": index, "op": node.op_type, "initializer": axes_name, "axes": axes})
    if not converted:
        raise RuntimeError("no constant Reduce axes")
    main_import.version = 17
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    uses.update(item.name for item in model.graph.output)
    removed: list[dict[str, object]] = []
    kept = []
    for initializer in model.graph.initializer:
        if uses[initializer.name] == 0:
            elements = int(np.prod(initializer.dims, dtype=np.int64)) if initializer.dims else 1
            removed.append({"initializer": initializer.name, "elements": elements})
        else:
            kept.append(initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model, {
        "original_opset": original_opset,
        "candidate_opset": 17,
        "converted": converted,
        "removed": removed,
        "parameter_saving": sum(int(row["elements"]) for row in removed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            try:
                candidate, details = build(model)
            except Exception as error:  # candidates that cannot safely downgrade are data
                if "no constant Reduce axes" not in str(error) and "below 18" not in str(error):
                    errors.append({"task": task, "error": f"{type(error).__name__}: {error}"})
                continue
            output = args.output_dir / f"task{task:03d}.onnx"
            onnx.save(candidate, output)
            rows.append({"task": task, "path": str(output), **details})
    result = {"candidates": rows, "errors": errors}
    (args.output_dir / "build_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
