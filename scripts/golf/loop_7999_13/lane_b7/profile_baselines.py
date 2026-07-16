#!/usr/bin/env python3
"""Measure exact B7 baselines and inventory their one-node tensor networks."""

from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TASKS = (30, 132, 175, 199, 212, 240, 304)


def main() -> None:
    rows: list[dict[str, object]] = []
    for task in TASKS:
        path = HERE / f"baseline_task{task:03d}.onnx"
        model = onnx.load(path)
        with tempfile.TemporaryDirectory(prefix=f"b7_base_{task:03d}_") as workdir:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = scoring.score_and_verify(
                    model, task, workdir, label="b7base", require_correct=True
                )
        if result is None:
            raise RuntimeError(f"baseline task{task:03d} did not score")
        initializers = []
        for init in model.graph.initializer:
            array = numpy_helper.to_array(init)
            initializers.append(
                {
                    "name": init.name,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "size": int(array.size),
                    "rank": int(np.linalg.matrix_rank(array)) if array.ndim == 2 else None,
                    "nonzero": int(np.count_nonzero(array)),
                }
            )
        node = model.graph.node[0]
        equation = next(
            (onnx.helper.get_attribute_value(attr).decode() for attr in node.attribute if attr.name == "equation"),
            None,
        )
        rows.append(
            {
                "task": task,
                "memory": int(result["memory"]),
                "params": int(result["params"]),
                "cost": int(result["cost"]),
                "known_correct": bool(result["correct"]),
                "nodes": len(model.graph.node),
                "ops": [item.op_type for item in model.graph.node],
                "equation": equation,
                "initializers": initializers,
            }
        )
    output = HERE / "baseline_profiles.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
