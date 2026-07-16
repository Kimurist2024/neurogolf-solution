#!/usr/bin/env python3
"""Measure exact baseline members and record graph/initializer inventory."""

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


TASKS = (75, 159, 200, 218, 225, 228, 388)


def main() -> None:
    rows: list[dict[str, object]] = []
    for task in TASKS:
        path = HERE / f"baseline_task{task:03d}.onnx"
        model = onnx.load(path)
        with tempfile.TemporaryDirectory(prefix=f"b6_base_{task:03d}_") as workdir:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = scoring.score_and_verify(
                    model, task, workdir, label="b6base", require_correct=True
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
                    "nonzero": int(np.count_nonzero(array)),
                    "finite": bool(np.all(np.isfinite(array))) if array.dtype.kind in "fc" else True,
                }
            )
        rows.append(
            {
                "task": task,
                "memory": int(result["memory"]),
                "params": int(result["params"]),
                "cost": int(result["cost"]),
                "known_correct": bool(result["correct"]),
                "nodes": len(model.graph.node),
                "initializers": initializers,
                "ops": [node.op_type for node in model.graph.node],
            }
        )
    output = HERE / "baseline_profiles.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
