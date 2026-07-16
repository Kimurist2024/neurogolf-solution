#!/usr/bin/env python3
"""Inventory shared-input Concat bases that may be factored inside Einsum."""

from __future__ import annotations

import hashlib
import itertools
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"


def attrs(node: onnx.NodeProto) -> dict[str, object]:
    return {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}


def main() -> None:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            name = f"task{task:03d}.onnx"
            data = archive.read(name)
            model = onnx.load_model_from_string(data)
            consumers: dict[str, list[tuple[int, str]]] = defaultdict(list)
            for index, node in enumerate(model.graph.node):
                for value in node.input:
                    consumers[value].append((index, node.op_type))
            concats = [
                (index, node)
                for index, node in enumerate(model.graph.node)
                if node.op_type == "Concat" and node.output
            ]
            for (left_index, left), (right_index, right) in itertools.combinations(concats, 2):
                left_axis = attrs(left).get("axis")
                right_axis = attrs(right).get("axis")
                if left_axis != right_axis:
                    continue
                shared = sorted(set(left.input) & set(right.input))
                if not shared:
                    continue
                left_users = consumers.get(left.output[0], [])
                right_users = consumers.get(right.output[0], [])
                einsum_only = bool(left_users and right_users) and all(
                    op == "Einsum" for _, op in left_users + right_users
                )
                rows.append(
                    {
                        "task": task,
                        "member_sha256": hashlib.sha256(data).hexdigest(),
                        "left_index": left_index,
                        "right_index": right_index,
                        "axis": left_axis,
                        "left_inputs": list(left.input),
                        "right_inputs": list(right.input),
                        "shared_inputs": shared,
                        "left_output": left.output[0],
                        "right_output": right.output[0],
                        "left_consumers": left_users,
                        "right_consumers": right_users,
                        "einsum_only": einsum_only,
                    }
                )
    result = {
        "authority": "submission_base_8009.46.zip",
        "authority_sha256": hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
        "pair_count": len(rows),
        "task_count": len({int(row["task"]) for row in rows}),
        "einsum_only_pair_count": sum(bool(row["einsum_only"]) for row in rows),
        "opportunity_tasks": sorted({int(row["task"]) for row in rows if row["einsum_only"]}),
        "rows": rows,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    for row in rows:
        print(
            f"task{int(row['task']):03d} pair={row['left_index']}/{row['right_index']} "
            f"shared={row['shared_inputs']} einsum_only={row['einsum_only']}"
        )


if __name__ == "__main__":
    main()
