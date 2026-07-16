#!/usr/bin/env python3
"""Find 2-D Einsum initializers reconstructible from another stored factor."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def equation(node: onnx.NodeProto) -> str | None:
    for attr in node.attribute:
        if attr.name == "equation":
            return attr.s.decode("ascii")
    return None


def candidate(target: np.ndarray, source: np.ndarray, side: str) -> tuple[np.ndarray, float, float]:
    if side == "right":
        # target[q,n] = transform[q,p] @ source[p,n]
        transform = target.astype(np.float64) @ np.linalg.pinv(source.astype(np.float64))
        transform32 = transform.astype(np.float32)
        rebuilt = transform32.astype(np.float64) @ source.astype(np.float64)
    else:
        # target[n,q] = source[n,p] @ transform[p,q]
        transform = np.linalg.pinv(source.astype(np.float64)) @ target.astype(np.float64)
        transform32 = transform.astype(np.float32)
        rebuilt = source.astype(np.float64) @ transform32.astype(np.float64)
    absolute = float(np.max(np.abs(rebuilt - target.astype(np.float64))))
    scale = max(1.0, float(np.max(np.abs(target))))
    return transform32, absolute, absolute / scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/golf/loop_7999_13/einsum_factor_reuse_audit.json"),
    )
    parser.add_argument("--max-relative-error", type=float, default=1e-5)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            initializers = {
                item.name: np.asarray(numpy_helper.to_array(item))
                for item in model.graph.initializer
            }
            uses = Counter(name for node in model.graph.node for name in node.input if name)
            for node_index, node in enumerate(model.graph.node):
                if node.op_type != "Einsum":
                    continue
                eq = equation(node)
                if not eq or "->" not in eq:
                    continue
                operands = eq.split("->", 1)[0].split(",")
                if len(operands) != len(node.input):
                    continue
                entries = []
                for input_index, (name, subscripts) in enumerate(zip(node.input, operands)):
                    array = initializers.get(name)
                    if array is not None and array.ndim == 2 and len(subscripts) == 2:
                        entries.append((input_index, name, subscripts, array))
                for target_index, target_name, target_subs, target in entries:
                    if uses[target_name] != sum(1 for name in node.input if name == target_name):
                        continue
                    for source_index, source_name, source_subs, source in entries:
                        if source_name == target_name:
                            continue
                        options: list[tuple[str, int]] = []
                        if target.shape[1] == source.shape[1] and target.shape[1] > source.shape[0]:
                            options.append(("right", target.shape[0] * source.shape[0]))
                        if target.shape[0] == source.shape[0] and target.shape[0] > source.shape[1]:
                            options.append(("left", source.shape[1] * target.shape[1]))
                        for side, transform_params in options:
                            saving = int(target.size - transform_params)
                            if saving <= 0:
                                continue
                            transform, absolute, relative = candidate(target, source, side)
                            if relative > args.max_relative_error:
                                continue
                            rows.append(
                                {
                                    "task": task,
                                    "node_index": node_index,
                                    "target_input_index": target_index,
                                    "target": target_name,
                                    "target_subscripts": target_subs,
                                    "target_shape": list(target.shape),
                                    "source_input_index": source_index,
                                    "source": source_name,
                                    "source_subscripts": source_subs,
                                    "source_shape": list(source.shape),
                                    "side": side,
                                    "transform_shape": list(transform.shape),
                                    "parameter_saving": saving,
                                    "max_abs_error": absolute,
                                    "max_relative_error": relative,
                                }
                            )
    rows.sort(
        key=lambda row: (
            -int(row["parameter_saving"]), float(row["max_relative_error"]), int(row["task"])
        )
    )
    result = {"source_zip": str(args.zip), "candidate_count": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
