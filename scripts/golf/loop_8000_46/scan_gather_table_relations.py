#!/usr/bin/env python3
"""Find exact algebraic relations among tables gathered with the same index.

If two initializer-backed Gather nodes use identical indices, a gathered
table can sometimes be derived from another gathered result.  That removes a
full initializer while replacing an already-counted Gather output with a
same-shaped arithmetic output, which can reduce official cost without
changing the executable input domain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def exact_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


def gather_axis(node: onnx.NodeProto) -> int:
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
    return int(attrs.get("axis", 0))


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
        groups: dict[tuple[str, int], list[dict[str, object]]] = {}
        for node_index, node in enumerate(model.graph.node):
            if node.op_type != "Gather" or len(node.input) < 2 or not node.output:
                continue
            table = node.input[0]
            if table not in arrays:
                continue
            key = (node.input[1], gather_axis(node))
            groups.setdefault(key, []).append(
                {
                    "node_index": node_index,
                    "table": table,
                    "output": node.output[0],
                    "array": arrays[table],
                }
            )

        for (indices, axis), rows in groups.items():
            if len(rows) < 2:
                continue
            for target in rows:
                target_array = target["array"]
                assert isinstance(target_array, np.ndarray)
                if target_array.size <= 1:
                    continue
                for base in rows:
                    if base is target:
                        continue
                    base_array = base["array"]
                    assert isinstance(base_array, np.ndarray)
                    if target_array.shape != base_array.shape or target_array.dtype != base_array.dtype:
                        continue
                    difference = target_array - base_array
                    scalar = difference.reshape(-1)[0]
                    if exact_equal(difference, np.full(difference.shape, scalar, dtype=difference.dtype)):
                        findings.append(
                            {
                                "task": int(path.stem.removeprefix("task")),
                                "kind": "base_plus_scalar",
                                "indices": indices,
                                "axis": axis,
                                "target_table": target["table"],
                                "target_output": target["output"],
                                "base_table": base["table"],
                                "base_output": base["output"],
                                "scalar": scalar.item(),
                                "table_elements": int(target_array.size),
                                "estimated_param_reduction": int(target_array.size - 1),
                            }
                        )
                for left in rows:
                    if left is target:
                        continue
                    left_array = left["array"]
                    assert isinstance(left_array, np.ndarray)
                    if target_array.shape != left_array.shape or target_array.dtype != left_array.dtype:
                        continue
                    for right in rows:
                        if right is target or right is left:
                            continue
                        right_array = right["array"]
                        assert isinstance(right_array, np.ndarray)
                        if target_array.shape != right_array.shape or target_array.dtype != right_array.dtype:
                            continue
                        for op, derived in (
                            ("add", left_array + right_array),
                            ("sub", left_array - right_array),
                        ):
                            if exact_equal(target_array, derived):
                                findings.append(
                                    {
                                        "task": int(path.stem.removeprefix("task")),
                                        "kind": op,
                                        "indices": indices,
                                        "axis": axis,
                                        "target_table": target["table"],
                                        "target_output": target["output"],
                                        "left_table": left["table"],
                                        "left_output": left["output"],
                                        "right_table": right["table"],
                                        "right_output": right["output"],
                                        "table_elements": int(target_array.size),
                                        "estimated_param_reduction": int(target_array.size),
                                    }
                                )
                            residual = target_array - derived
                            scalar = residual.reshape(-1)[0]
                            if exact_equal(
                                residual,
                                np.full(residual.shape, scalar, dtype=residual.dtype),
                            ):
                                findings.append(
                                    {
                                        "task": int(path.stem.removeprefix("task")),
                                        "kind": f"{op}_plus_scalar",
                                        "indices": indices,
                                        "axis": axis,
                                        "target_table": target["table"],
                                        "target_output": target["output"],
                                        "left_table": left["table"],
                                        "left_output": left["output"],
                                        "right_table": right["table"],
                                        "right_output": right["output"],
                                        "scalar": scalar.item(),
                                        "table_elements": int(target_array.size),
                                        "estimated_param_reduction": int(target_array.size - 1),
                                    }
                                )

    unique = {
        json.dumps(item, sort_keys=True, allow_nan=False): item
        for item in findings
    }
    ranked = sorted(
        unique.values(),
        key=lambda item: (-int(item["estimated_param_reduction"]), int(item["task"])),
    )
    payload = {
        "models": str(args.models),
        "finding_count": len(ranked),
        "findings": ranked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
