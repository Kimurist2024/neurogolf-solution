#!/usr/bin/env python3
"""Find opset>=18 models where constant Reduce axes can become attributes."""

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


REDUCE_OPS = {
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/golf/loop_7999_13/opset_axes_downgrade_audit.json"),
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    distribution: Counter[int] = Counter()
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            main_import = next((item for item in model.opset_import if not item.domain), None)
            if main_import is None:
                continue
            opset = int(main_import.version)
            distribution[opset] += 1
            if opset < 18:
                continue
            inits = {item.name: item for item in model.graph.initializer}
            uses = Counter(name for node in model.graph.node for name in node.input if name)
            conversions: list[dict[str, object]] = []
            unique_names: set[str] = set()
            for index, node in enumerate(model.graph.node):
                if node.op_type not in REDUCE_OPS or len(node.input) < 2 or not node.input[1]:
                    continue
                axes = inits.get(node.input[1])
                if axes is None:
                    continue
                values = np.asarray(numpy_helper.to_array(axes), dtype=np.int64).reshape(-1)
                conversions.append(
                    {
                        "node_index": index,
                        "op": node.op_type,
                        "initializer": axes.name,
                        "axes": values.tolist(),
                        "unique_use": uses[axes.name] == 1,
                    }
                )
                if uses[axes.name] == 1:
                    unique_names.add(axes.name)
            if conversions:
                saving = sum(
                    int(np.prod(inits[name].dims, dtype=np.int64)) if inits[name].dims else 1
                    for name in unique_names
                )
                rows.append(
                    {
                        "task": task,
                        "opset": opset,
                        "conversions": conversions,
                        "potential_parameter_saving": saving,
                    }
                )
    rows.sort(key=lambda row: (-int(row["potential_parameter_saving"]), int(row["task"])))
    result = {
        "source_zip": str(args.zip),
        "opset_distribution": dict(sorted(distribution.items())),
        "candidate_task_count": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
