#!/usr/bin/env python3
"""Verify the generator-derived reference rules for both C25 tasks."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def import_path(label: str, path: Path):
    spec = importlib.util.spec_from_file_location(label, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_task131(n: int) -> dict[str, object]:
    ref = import_path("c25_task131_ref", ROOT / "scripts/golf/scratch_codex/task131/task131_reference.py")
    data = json.loads((ROOT / "inputs/neurogolf-2026/task131.json").read_text(encoding="utf-8"))
    visible = 0
    for subset in ("train", "test", "arc-gen"):
        for index, example in enumerate(data[subset], 1):
            assert ref.solve(example["input"]) == example["output"], f"task131 {subset}[{index}]"
            visible += 1
    generator = ref._load_generator()
    for seed in range(n):
        random.seed(seed)
        example = generator.generate()
        assert ref.solve(example["input"]) == example["output"], f"task131 fresh seed {seed}"
    return {
        "task": 131,
        "reference": "scripts/golf/scratch_codex/task131/task131_reference.py::solve",
        "rule": "move the green object adjacent to the red line and draw the cyan separator one cell beyond it",
        "visible": {"right": visible, "wrong": 0, "errors": 0},
        "fresh": {"seeds": [0, n - 1], "right": n, "wrong": 0, "errors": 0},
    }


def verify_task251(n: int) -> dict[str, object]:
    ref = import_path("c25_task251_ref", ROOT / "scripts/golf/scratch_codex/task251/task251_ref.py")
    data = json.loads((ROOT / "inputs/neurogolf-2026/task251.json").read_text(encoding="utf-8"))
    visible = 0
    for subset in ("train", "test", "arc-gen"):
        for index, example in enumerate(data[subset], 1):
            assert ref.enclosed_solver(example["input"]) == example["output"], f"task251 {subset}[{index}]"
            visible += 1
    generator = ref.load_generator()
    for seed in range(n):
        random.seed(seed)
        example = generator.generate()
        assert ref.enclosed_solver(example["input"]) == example["output"], f"task251 fresh seed {seed}"
    return {
        "task": 251,
        "reference": "scripts/golf/scratch_codex/task251/task251_ref.py::enclosed_solver",
        "rule": "paint black cells enclosed from the grid boundary blue",
        "visible": {"right": visible, "wrong": 0, "errors": 0},
        "fresh": {"seeds": [0, n - 1], "right": n, "wrong": 0, "errors": 0},
    }


def main() -> None:
    n = 5000
    output = {
        "generator_truth": [verify_task131(n), verify_task251(n)],
        "perfect": True,
    }
    (HERE / "reference_verification.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    for row in output["generator_truth"]:
        print(f"task{row['task']}: visible={row['visible']['right']} fresh={row['fresh']['right']}")


if __name__ == "__main__":
    main()
