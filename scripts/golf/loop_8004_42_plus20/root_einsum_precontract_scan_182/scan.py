#!/usr/bin/env python3
"""Find parameter-lowering two-initializer precontractions in Einsum nodes."""

from __future__ import annotations

import hashlib
import itertools
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"


def equation(node: onnx.NodeProto) -> str | None:
    for attr in node.attribute:
        if attr.name == "equation":
            value = onnx.helper.get_attribute_value(attr)
            return value.decode() if isinstance(value, bytes) else str(value)
    return None


def main() -> None:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            member = f"task{task:03d}.onnx"
            data = archive.read(member)
            model = onnx.load_model_from_string(data)
            initializers = {item.name: item for item in model.graph.initializer}
            uses: Counter[str] = Counter(value for node in model.graph.node for value in node.input)
            for node_index, node in enumerate(model.graph.node):
                if node.op_type != "Einsum":
                    continue
                eq = equation(node)
                if not eq or "..." in eq or "->" not in eq:
                    continue
                left, output_labels = eq.split("->", 1)
                terms = left.split(",")
                if len(terms) != len(node.input) or any(len(set(term)) != len(term) for term in terms):
                    continue
                label_occurrences = Counter(label for term in terms for label in term)
                initializer_positions = [
                    index for index, value in enumerate(node.input) if value in initializers
                ]
                for left_pos, right_pos in itertools.combinations(initializer_positions, 2):
                    left_labels, right_labels = terms[left_pos], terms[right_pos]
                    shared = set(left_labels) & set(right_labels)
                    contracted = sorted(
                        label
                        for label in shared
                        if label not in output_labels and label_occurrences[label] == 2
                    )
                    if not contracted:
                        continue
                    # Preserve first appearance while removing contracted labels.
                    result_labels = "".join(
                        dict.fromkeys(
                            label
                            for label in left_labels + right_labels
                            if label not in contracted
                        )
                    )
                    left_tensor = initializers[node.input[left_pos]]
                    right_tensor = initializers[node.input[right_pos]]
                    left_array = numpy_helper.to_array(left_tensor)
                    right_array = numpy_helper.to_array(right_tensor)
                    try:
                        combined = np.einsum(
                            f"{left_labels},{right_labels}->{result_labels}",
                            left_array,
                            right_array,
                        )
                    except Exception:
                        continue
                    before = int(left_array.size + right_array.size)
                    after = int(combined.size)
                    if after >= before:
                        continue
                    rows.append(
                        {
                            "task": task,
                            "member_sha256": hashlib.sha256(data).hexdigest(),
                            "node_index": node_index,
                            "equation": eq,
                            "left_position": left_pos,
                            "right_position": right_pos,
                            "left_name": node.input[left_pos],
                            "right_name": node.input[right_pos],
                            "left_labels": left_labels,
                            "right_labels": right_labels,
                            "contracted_labels": contracted,
                            "result_labels": result_labels,
                            "before_elements": before,
                            "after_elements": after,
                            "element_saving": before - after,
                            "left_graph_uses": uses[node.input[left_pos]],
                            "right_graph_uses": uses[node.input[right_pos]],
                            "left_dtype": str(left_array.dtype),
                            "right_dtype": str(right_array.dtype),
                            "finite": bool(np.isfinite(left_array).all() and np.isfinite(right_array).all()),
                            "integer_valued": bool(
                                np.equal(left_array, np.trunc(left_array)).all()
                                and np.equal(right_array, np.trunc(right_array)).all()
                            ),
                            "combined_max_abs": float(np.max(np.abs(combined))) if combined.size else 0.0,
                        }
                    )
    rows.sort(key=lambda row: (-int(row["element_saving"]), int(row["task"])))
    result = {
        "authority": "submission_base_8009.46.zip",
        "authority_sha256": hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
        "candidate_pair_count": len(rows),
        "task_count": len({int(row["task"]) for row in rows}),
        "integer_pair_count": sum(bool(row["integer_valued"]) for row in rows),
        "single_use_pair_count": sum(
            row["left_graph_uses"] == 1 and row["right_graph_uses"] == 1 for row in rows
        ),
        "rows": rows,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    for row in rows[:100]:
        print(
            f"task{int(row['task']):03d} node{row['node_index']} "
            f"{row['left_name']}[{row['left_labels']}] x {row['right_name']}[{row['right_labels']}] "
            f"->[{row['result_labels']}] save={row['element_saving']} "
            f"uses={row['left_graph_uses']}/{row['right_graph_uses']} integer={row['integer_valued']}"
        )


if __name__ == "__main__":
    main()
