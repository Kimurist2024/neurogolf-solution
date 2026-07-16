#!/usr/bin/env python3
"""Persist constructive non-injectivity witnesses for B16 generators."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(TASK_DIR))


def digest(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def task157() -> dict[str, Any]:
    module = importlib.import_module("task_6a1e5592")
    shapes = [
        [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0)],
    ]
    rows: list[int] = []
    cols: list[int] = []
    idxs: list[int] = []
    for index, shape in enumerate(shapes):
        for row, col in shape:
            rows.append(row)
            cols.append(col)
            idxs.append(index)
    common = {
        "rows": rows,
        "cols": cols,
        "idxs": idxs,
        "bluerows": [2, 2, 2],
        "grayrows": [6, 8, 8],
        "graycols": [7, 1, 11],
    }
    first = module.generate(bluecols=[2, 11, 0], **common)
    second = module.generate(bluecols=[0, 11, 4], **common)
    assert first["input"] == second["input"]
    assert first["output"] != second["output"]
    return {
        "hash": "6a1e5592",
        "same_input": True,
        "different_output": True,
        "input_sha256": digest(first["input"]),
        "output_a_sha256": digest(first["output"]),
        "output_b_sha256": digest(second["output"]),
        "bluecols_a": [2, 11, 0],
        "bluecols_b": [0, 11, 4],
    }


def task319() -> dict[str, Any]:
    module = importlib.import_module("task_ce602527")
    common = {
        "width": 15,
        "height": 18,
        "rows": [0, 1, 2, 2, 0, 1, 2, 2],
        "cols": [1, 2, 0, 1, 1, 2, 0, 1],
        "idxs": [0, 0, 0, 0, 1, 1, 1, 1],
        "magrow": 5,
        "magcol": 11,
        "magcolor": 2,
        "bgcolor": 9,
    }
    first = module.generate(brows=[5, 0], bcols=[1, 7], colors=[5, 4], **common)
    second = module.generate(brows=[0, 5], bcols=[7, 1], colors=[4, 5], **common)
    assert first["input"] == second["input"]
    assert first["output"] != second["output"]
    return {
        "hash": "ce602527",
        "same_input": True,
        "different_output": True,
        "input_sha256": digest(first["input"]),
        "output_a_sha256": digest(first["output"]),
        "output_b_sha256": digest(second["output"]),
        "output_a": first["output"],
        "output_b": second["output"],
    }


def main() -> int:
    report = {
        "conclusion": (
            "Both generators admit identical observable inputs with distinct gold outputs; "
            "zero-error deterministic ONNX coverage of every valid call is impossible."
        ),
        "tasks": {"157": task157(), "319": task319()},
    }
    (HERE / "generator_collision_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
