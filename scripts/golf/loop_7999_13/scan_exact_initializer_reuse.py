#!/usr/bin/env python3
"""Find exact initializer reuse and dead-parameter reductions in the 7999.13 ZIP.

This is deliberately a read-only audit.  It reports only transformations whose
tensor bytes, dtype, and shape prove algebraic identity; candidate construction
and strict runtime validation are separate steps.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def tensor_key(initializer: onnx.TensorProto) -> tuple[int, tuple[int, ...], bytes]:
    array = np.ascontiguousarray(numpy_helper.to_array(initializer))
    # np.ascontiguousarray promotes a rank-0 ndarray to shape (1,), so the
    # serialized ONNX dims—not the converted ndarray shape—are authoritative.
    return int(initializer.data_type), tuple(int(x) for x in initializer.dims), array.tobytes()


def all_consumed_names(model: onnx.ModelProto) -> set[str]:
    names: set[str] = set()
    for node in model.graph.node:
        names.update(name for name in node.input if name)
    names.update(output.name for output in model.graph.output)
    return names


def audit_model(task: int, model: onnx.ModelProto) -> dict[str, object]:
    consumed = all_consumed_names(model)
    initializers = list(model.graph.initializer)
    dead = [item.name for item in initializers if item.name not in consumed]

    exact_groups: dict[tuple[int, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for item in initializers:
        exact_groups[tensor_key(item)].append(item.name)
    duplicates = [names for names in exact_groups.values() if len(names) > 1]

    scalar_groups: dict[tuple[int, bytes], list[str]] = defaultdict(list)
    for item in initializers:
        array = np.ascontiguousarray(numpy_helper.to_array(item))
        if array.size == 1:
            scalar_groups[(int(item.data_type), array.tobytes())].append(item.name)
    duplicate_scalars = [names for names in scalar_groups.values() if len(names) > 1]

    dead_elements = 0
    for item in initializers:
        if item.name in dead:
            dead_elements += int(np.prod(item.dims, dtype=np.int64)) if item.dims else 1
    duplicate_elements = 0
    for key, names in exact_groups.items():
        if len(names) > 1:
            shape = key[1]
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            duplicate_elements += (len(names) - 1) * elements

    return {
        "task": task,
        "initializer_count": len(initializers),
        "dead": dead,
        "dead_elements": dead_elements,
        "exact_duplicate_groups": duplicates,
        "exact_duplicate_elements": duplicate_elements,
        "duplicate_scalar_groups": duplicate_scalars,
        "potential_parameter_reduction": dead_elements + duplicate_elements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/golf/loop_7999_13/exact_initializer_reuse_audit.json"),
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            payload = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_model(io.BytesIO(payload))
            row = audit_model(task, model)
            row["sha256"] = hashlib.sha256(payload).hexdigest()
            if row["potential_parameter_reduction"] or row["duplicate_scalar_groups"]:
                rows.append(row)

    document = {
        "source_zip": str(args.zip),
        "candidate_task_count": len(rows),
        "rows": sorted(
            rows,
            key=lambda row: (-int(row["potential_parameter_reduction"]), int(row["task"])),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
