#!/usr/bin/env python3
"""Find initializers that are exact axis slices of another stored initializer."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/golf/loop_7999_13/initializer_slice_reuse_audit.json"),
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            tensors = {
                item.name: (item, np.asarray(numpy_helper.to_array(item)))
                for item in model.graph.initializer
            }
            uses: dict[str, list[dict[str, object]]] = {name: [] for name in tensors}
            for node_index, node in enumerate(model.graph.node):
                for input_index, name in enumerate(node.input):
                    if name in uses:
                        uses[name].append(
                            {"node_index": node_index, "input_index": input_index, "op": node.op_type}
                        )
            for small_name, (small_proto, small) in tensors.items():
                if small.ndim == 0 or not uses[small_name]:
                    continue
                if any(use["op"] != "Einsum" for use in uses[small_name]):
                    continue
                for large_name, (large_proto, large) in tensors.items():
                    if large_name == small_name or large.dtype != small.dtype:
                        continue
                    if large.ndim != small.ndim + 1 or large.size <= small.size:
                        continue
                    for axis in range(large.ndim):
                        expected_shape = large.shape[:axis] + large.shape[axis + 1 :]
                        if expected_shape != small.shape:
                            continue
                        selector_params = int(large.shape[axis])
                        saving = int(small.size - selector_params)
                        if saving <= 0:
                            continue
                        for index in range(large.shape[axis]):
                            if np.array_equal(np.take(large, index, axis=axis), small):
                                rows.append(
                                    {
                                        "task": task,
                                        "small": small_name,
                                        "small_shape": list(small.shape),
                                        "large": large_name,
                                        "large_shape": list(large.shape),
                                        "axis": axis,
                                        "index": index,
                                        "uses": uses[small_name],
                                        "selector_params": selector_params,
                                        "potential_parameter_saving": saving,
                                    }
                                )
    rows.sort(key=lambda row: (-int(row["potential_parameter_saving"]), int(row["task"])))
    result = {"source_zip": str(args.zip), "candidate_count": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
