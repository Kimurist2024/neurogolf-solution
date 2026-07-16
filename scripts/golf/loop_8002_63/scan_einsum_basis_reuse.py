#!/usr/bin/env python3
"""Scan the latest LB models for exact in-Einsum initializer basis reuse.

The useful pattern is ``target = transform @ source`` (or its right-sided
analogue), where target and source are already operands of the same Einsum and
the transform has fewer elements than target.  The replacement can stay
inside the existing Einsum, so it need not materialize a scored activation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def exact_left_transform(target: np.ndarray, source: np.ndarray) -> np.ndarray | None:
    if target.ndim != 2 or source.ndim != 2 or target.shape[1] != source.shape[1]:
        return None
    target_rows, cols = target.shape
    source_rows = source.shape[0]
    if target_rows * source_rows >= target.size:
        return None
    try:
        solved = target.astype(np.float64) @ np.linalg.pinv(source.astype(np.float64))
    except np.linalg.LinAlgError:
        return None
    transform = solved.astype(target.dtype)
    rebuilt = np.matmul(transform, source).astype(target.dtype)
    return transform if np.array_equal(rebuilt, target, equal_nan=True) else None


def proportional(target: np.ndarray, source: np.ndarray) -> float | None:
    if target.shape != source.shape or target.dtype != source.dtype:
        return None
    if not np.array_equal(target == 0, source == 0):
        return None
    mask = source != 0
    if not mask.any():
        return None
    ratios = target[mask].astype(np.float64) / source[mask].astype(np.float64)
    ratio = ratios[0]
    rebuilt = (source.astype(np.float64) * ratio).astype(target.dtype)
    return float(ratio) if np.array_equal(rebuilt, target, equal_nan=True) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    findings: list[dict[str, object]] = []
    for path in sorted(args.models.glob("task*.onnx")):
        model = onnx.load(path, load_external_data=False)
        arrays = {
            item.name: np.asarray(numpy_helper.to_array(item))
            for item in model.graph.initializer
        }
        for node_index, node in enumerate(model.graph.node):
            if node.op_type != "Einsum" or not node.attribute:
                continue
            equation_attr = next((item for item in node.attribute if item.name == "equation"), None)
            if equation_attr is None:
                continue
            equation = equation_attr.s.decode("utf-8")
            input_equations = equation.split("->", 1)[0].split(",")
            operands = [
                (position, name, arrays[name], input_equations[position])
                for position, name in enumerate(node.input)
                if name in arrays and position < len(input_equations)
            ]
            for target, source in itertools.permutations(operands, 2):
                target_pos, target_name, target_array, target_eq = target
                source_pos, source_name, source_array, source_eq = source
                if target_name == source_name:
                    continue
                if target_array.dtype.kind not in "fc" or source_array.dtype != target_array.dtype:
                    continue

                ratio = proportional(target_array, source_array)
                if ratio is not None and target_array.size > 1:
                    findings.append(
                        {
                            "task": int(path.stem.removeprefix("task")),
                            "node_index": node_index,
                            "equation": equation,
                            "kind": "proportional",
                            "target": target_name,
                            "target_position": target_pos,
                            "target_subscript": target_eq,
                            "source": source_name,
                            "source_position": source_pos,
                            "source_subscript": source_eq,
                            "ratio": ratio,
                            "target_elements": int(target_array.size),
                        }
                    )

                left = exact_left_transform(target_array, source_array)
                if left is not None:
                    findings.append(
                        {
                            "task": int(path.stem.removeprefix("task")),
                            "node_index": node_index,
                            "equation": equation,
                            "kind": "left_basis",
                            "target": target_name,
                            "target_position": target_pos,
                            "target_subscript": target_eq,
                            "target_shape": list(target_array.shape),
                            "source": source_name,
                            "source_position": source_pos,
                            "source_subscript": source_eq,
                            "source_shape": list(source_array.shape),
                            "transform_shape": list(left.shape),
                            "transform": left.tolist(),
                            "target_elements": int(target_array.size),
                            "transform_elements": int(left.size),
                            "estimated_param_reduction": int(target_array.size - left.size),
                        }
                    )

                right_t = exact_left_transform(target_array.T, source_array.T)
                if right_t is not None:
                    right = right_t.T
                    findings.append(
                        {
                            "task": int(path.stem.removeprefix("task")),
                            "node_index": node_index,
                            "equation": equation,
                            "kind": "right_basis",
                            "target": target_name,
                            "target_position": target_pos,
                            "target_subscript": target_eq,
                            "target_shape": list(target_array.shape),
                            "source": source_name,
                            "source_position": source_pos,
                            "source_subscript": source_eq,
                            "source_shape": list(source_array.shape),
                            "transform_shape": list(right.shape),
                            "transform": right.tolist(),
                            "target_elements": int(target_array.size),
                            "transform_elements": int(right.size),
                            "estimated_param_reduction": int(target_array.size - right.size),
                        }
                    )

    unique = {
        json.dumps(item, sort_keys=True, allow_nan=False): item
        for item in findings
    }
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            -int(item.get("estimated_param_reduction", 0)),
            -int(item.get("target_elements", 0)),
            int(item["task"]),
        ),
    )
    payload = {
        "models": str(args.models),
        "finding_count": len(ranked),
        "findings": ranked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "models": payload["models"],
                "finding_count": payload["finding_count"],
                "top_findings": ranked[:40],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
