#!/usr/bin/env python3
"""Find higher-rank initializer tensors reusable through one mode transform."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def equation(node: onnx.NodeProto) -> str | None:
    for attr in node.attribute:
        if attr.name == "equation":
            return attr.s.decode("ascii")
    return None


def unfold(array: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(array, axis, 0).reshape(array.shape[axis], -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument("--max-relative-error", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/golf/loop_7999_13/tensor_mode_reuse_audit.json"),
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            arrays = {
                item.name: np.asarray(numpy_helper.to_array(item))
                for item in model.graph.initializer
                if len(item.dims) >= 3
            }
            uses: dict[str, list[dict[str, object]]] = {name: [] for name in arrays}
            invalid: set[str] = set()
            for node_index, node in enumerate(model.graph.node):
                eq = equation(node) if node.op_type == "Einsum" else None
                operands = eq.split("->", 1)[0].split(",") if eq and "->" in eq else []
                for input_index, name in enumerate(node.input):
                    if name not in arrays:
                        continue
                    if (
                        node.op_type != "Einsum"
                        or input_index >= len(operands)
                        or len(operands[input_index]) != arrays[name].ndim
                    ):
                        invalid.add(name)
                    else:
                        uses[name].append(
                            {
                                "node_index": node_index,
                                "input_index": input_index,
                                "subscripts": operands[input_index],
                            }
                        )
            for target_name, target in arrays.items():
                if target_name in invalid or not uses[target_name]:
                    continue
                for source_name, source in arrays.items():
                    if source_name == target_name or source.ndim != target.ndim or source.dtype != target.dtype:
                        continue
                    for axis in range(target.ndim):
                        if any(
                            target.shape[index] != source.shape[index]
                            for index in range(target.ndim)
                            if index != axis
                        ):
                            continue
                        target_matrix = unfold(target, axis).astype(np.float64)
                        source_matrix = unfold(source, axis).astype(np.float64)
                        transform64 = target_matrix @ np.linalg.pinv(source_matrix)
                        transform = transform64.astype(target.dtype)
                        saving = int(target.size - transform.size)
                        if saving <= 0:
                            continue
                        rebuilt_matrix = transform.astype(np.float64) @ source_matrix
                        absolute = float(np.max(np.abs(rebuilt_matrix - target_matrix)))
                        relative = absolute / max(1.0, float(np.max(np.abs(target_matrix))))
                        if relative <= args.max_relative_error:
                            rows.append(
                                {
                                    "task": task,
                                    "target": target_name,
                                    "target_shape": list(target.shape),
                                    "source": source_name,
                                    "source_shape": list(source.shape),
                                    "axis": axis,
                                    "transform_shape": list(transform.shape),
                                    "parameter_saving": saving,
                                    "max_abs_error": absolute,
                                    "max_relative_error": relative,
                                    "uses": uses[target_name],
                                }
                            )
    rows.sort(key=lambda row: (-int(row["parameter_saving"]), float(row["max_relative_error"])))
    result = {"source_zip": str(args.zip), "candidate_count": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
