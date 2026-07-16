#!/usr/bin/env python3
"""Two independent fresh-5000 audits in default and disabled ORT modes."""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(ROOT / "scripts/golf/scratch_codex/task216"))

from lib import scoring  # noqa: E402
from reference import solve_grid as solve_216  # noqa: E402


COUNT = 5000
SEEDS = (2162551031, 2162551032)
GENERATORS = {216: "task_8efcae92", 255: "task_a64e4611"}


def encode(grid: list[list[int]]) -> np.ndarray:
    value = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            value[0, color, row, col] = 1.0
    return value


def boxes_overlap(
    brows: list[int],
    bcols: list[int],
    wides: list[int],
    talls: list[int],
) -> bool:
    """Mirror common.overlaps(..., spacing=1)."""
    for j in range(len(brows)):
        for i in range(j):
            if brows[i] + talls[i] + 1 <= brows[j]:
                continue
            if brows[j] + talls[j] + 1 <= brows[i]:
                continue
            if bcols[i] + wides[i] + 1 <= bcols[j]:
                continue
            if bcols[j] + wides[j] + 1 <= bcols[i]:
                continue
            return True
    return False


def fresh_216(generator: Any, rng: random.Random, index: int) -> dict[str, Any]:
    """Construct a legal generator instance without its slow placement rejection.

    Every parameter tuple is inside task_8efcae92.generate's support.  The six
    layout families deliberately include stacked equal-left-edge boxes and the
    maximal legal top/left starts that compact historical graphs missed.
    """
    n = 3 + (index & 1)
    mode = index % 6
    if mode == 0:
        starts = [0, 4, 8, 17][:n]
        rng.shuffle(starts)
        brows = starts
        bcols = [rng.randint(0, 2)] * n
        wides = [rng.randint(4, 18 - bcols[0]) for _ in range(n)]
        talls = [3] * n
    elif mode == 1:
        if n == 4:
            starts, widths = [0, 5, 10, 15], [4, 4, 4, 5]
        else:
            starts, widths = [0, 6, 12], [5, 5, 8]
        order = list(range(n))
        rng.shuffle(order)
        bcols = [starts[i] for i in order]
        wides = [widths[i] for i in order]
        brows = [rng.randint(0, 2)] * n
        talls = [rng.randint(3, 18 - brows[0]) for _ in range(n)]
    elif mode == 2:
        cells = [(0, 0), (0, 11), (11, 0), (11, 11)][:n]
        rng.shuffle(cells)
        brows = [cell[0] for cell in cells]
        bcols = [cell[1] for cell in cells]
        wides = [rng.randint(4, 8 if col == 11 else 10) for col in bcols]
        talls = [rng.randint(3, 8 if row == 11 else 10) for row in brows]
    elif mode == 3:
        brows = [17, 0, 4, 8][:n]
        bcols = [rng.randint(0, 2)] * n
        wides = [rng.randint(4, 18 - bcols[0]) for _ in range(n)]
        talls = [3] * n
    elif mode == 4:
        bcols = [16, 0, 5, 10][:n]
        wides = [4, 4, 4, 5][:n]
        brows = [rng.randint(0, 2)] * n
        talls = [rng.randint(3, 18 - brows[0]) for _ in range(n)]
    else:
        while True:
            brows, bcols, wides, talls = [], [], [], []
            for _ in range(n):
                placed = False
                for _ in range(200):
                    wide = rng.randint(4, 10)
                    tall = rng.randint(3, 10)
                    row = rng.randint(0, 20 - tall)
                    col = rng.randint(0, 20 - wide)
                    if not boxes_overlap(
                        brows + [row],
                        bcols + [col],
                        wides + [wide],
                        talls + [tall],
                    ):
                        brows.append(row)
                        bcols.append(col)
                        wides.append(wide)
                        talls.append(tall)
                        placed = True
                        break
                if not placed:
                    break
            if len(brows) == n:
                break
    assert not boxes_overlap(brows, bcols, wides, talls)
    min_area = min(w * h for w, h in zip(wides, talls))
    maximum = rng.randint(n, min(min_area, 24))
    counts = [maximum - i for i in range(n)]
    rows: list[int] = []
    cols: list[int] = []
    idxs: list[int] = []
    for box, (wide, tall, count) in enumerate(zip(wides, talls, counts)):
        for flat in rng.sample(range(wide * tall), count):
            rows.append(flat // wide)
            cols.append(flat % wide)
            idxs.append(box)
    return generator.generate(
        rows=rows,
        cols=cols,
        idxs=idxs,
        brows=brows,
        bcols=bcols,
        wides=wides,
        talls=talls,
        size=20,
    )


def make_session(task: int, mode: str) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(HERE / f"task{task:03d}.onnx")))
    if model is None:
        raise RuntimeError(f"task{task}: sanitize failed")
    options = ort.SessionOptions()
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options)


def empty_row(task: int, seed: int, mode: str) -> dict[str, Any]:
    return {
        "task": task,
        "seed": seed,
        "count": COUNT,
        "mode": mode,
        "right": 0,
        "wrong": 0,
        "errors": 0,
        "nonfinite": 0,
        "mid_margin": 0,
        "min_positive": None,
        "first_failure": None,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    generators = {
        task: importlib.import_module(module) for task, module in GENERATORS.items()
    }
    sessions = {
        (task, mode): make_session(task, mode)
        for task in GENERATORS
        for mode in ("disabled", "default")
    }
    rows: list[dict[str, Any]] = []
    reference_216: list[dict[str, int]] = []
    for task, generator in generators.items():
        for seed in SEEDS:
            random.seed(seed + task)
            rng = random.Random(seed + task)
            task_rows = {
                mode: empty_row(task, seed, mode)
                for mode in ("disabled", "default")
            }
            ref_right = ref_wrong = 0
            for index in range(COUNT):
                example = (
                    fresh_216(generator, rng, index)
                    if task == 216
                    else generator.generate()
                )
                input_value = encode(example["input"])
                expected = encode(example["output"]).astype(bool)
                if task == 216:
                    ref = encode(solve_216(example["input"])).astype(bool)
                    if np.array_equal(ref, expected):
                        ref_right += 1
                    else:
                        ref_wrong += 1
                for mode, row in task_rows.items():
                    try:
                        raw = np.asarray(
                            sessions[(task, mode)].run(
                                ["output"], {"input": input_value}
                            )[0]
                        )
                        finite = np.isfinite(raw)
                        row["nonfinite"] += int(np.count_nonzero(~finite))
                        row["mid_margin"] += int(
                            np.count_nonzero(finite & (raw > 0.0) & (raw < 0.25))
                        )
                        positive = raw[finite & (raw > 0.0)]
                        if positive.size:
                            current = float(positive.min())
                            prior = row["min_positive"]
                            row["min_positive"] = (
                                current if prior is None else min(prior, current)
                            )
                        if np.array_equal(raw > 0.0, expected):
                            row["right"] += 1
                        else:
                            row["wrong"] += 1
                            row["first_failure"] = row["first_failure"] or {
                                "index": index,
                                "kind": "wrong_output",
                            }
                    except Exception as exc:  # noqa: BLE001
                        row["errors"] += 1
                        row["first_failure"] = row["first_failure"] or {
                            "index": index,
                            "kind": "runtime_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                if (index + 1) % 1000 == 0:
                    print(task, seed, "progress", index + 1, flush=True)
            for row in task_rows.values():
                row["accuracy"] = row["right"] / COUNT
                row["pass_exact"] = (
                    row["right"] == COUNT
                    and row["wrong"] == 0
                    and row["errors"] == 0
                    and row["nonfinite"] == 0
                    and row["mid_margin"] == 0
                )
                rows.append(row)
                print(row, flush=True)
            if task == 216:
                reference_216.append(
                    {
                        "seed": seed,
                        "count": COUNT,
                        "right": ref_right,
                        "wrong": ref_wrong,
                    }
                )
    report = {
        "count_per_seed": COUNT,
        "seeds": list(SEEDS),
        "task216_sampling": (
            "legal explicit task_8efcae92.generate parameters; six layout families, "
            "including stacked same-left-edge and maximal row/col starts"
        ),
        "task255_sampling": "native task_a64e4611.generate rejection-free path",
        "rows": rows,
        "task216_reference": reference_216,
    }
    (HERE / "fresh_two_seeds.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
