#!/usr/bin/env python3
"""Generator-derived NumPy reference audit for tasks 018/233/286/366.

Three transforms are deterministic and are checked for exact equality.  Task018
is intentionally reported separately: task_0e206a2e is non-injective (the same
visible marker input can correspond to multiple rotations and outputs), so no
deterministic input-only reference or ONNX can be exact on the unrestricted
generator.  Its spec-derived canonical solver is measured rather than called
"exact".
"""

from __future__ import annotations

import importlib
import importlib.util
import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo"))
sys.path.insert(0, str(TASKS_DIR))


def load_source(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# These are the repository's retained spec-derived NumPy sources.  This lane
# pins their hashes in its output so the evidence cannot silently drift.
REFS = {
    18: load_source("high3_ref018", ROOT / "scripts/golf/scratch/task018/staticref2.py"),
    233: load_source("high3_ref233", ROOT / "scripts/golf/scratch/task233/ref_final.py"),
    286: load_source("high3_ref286", ROOT / "scripts/golf/scratch/task286/ref_clean.py"),
    366: load_source("high3_ref366", ROOT / "scripts/golf/scratch/task366/ref7.py"),
}
HASHES = {
    task: __import__("hashlib").sha256(Path(module.__file__).read_bytes()).hexdigest()
    for task, module in REFS.items()
}
GENERATORS = {
    18: importlib.import_module("task_0e206a2e"),
    233: importlib.import_module("task_97a05b5b"),
    286: importlib.import_module("task_b782dc8a"),
    366: importlib.import_module("task_e6721834"),
}


def solve(task: int, grid: list[list[int]]) -> np.ndarray | None:
    array = np.asarray(grid, dtype=np.int64)
    if task == 286:
        return np.asarray(REFS[task].transform(grid), dtype=np.int64)
    return REFS[task].solve(array)


def _components(mask: np.ndarray, diagonal: bool = False) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    steps = [
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)
    ]
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            seen[row, col] = True
            stack = [(row, col)]
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in steps:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            result.append(cells)
    return result


def solve233_generated(grid: list[list[int]]) -> np.ndarray:
    """Exact fast path for generate() cases (the generator fixes rotates=[0])."""
    source = np.asarray(grid, dtype=np.int64)
    height, width = source.shape
    red_components = _components(source == 2)
    red_box = max(red_components, key=len)
    rows = [cell[0] for cell in red_box]
    cols = [cell[1] for cell in red_box]
    brow, bend, bcol, cend = min(rows), max(rows), min(cols), max(cols)
    box_mask = np.zeros((height, width), dtype=bool)
    box_mask[brow : bend + 1, bcol : cend + 1] = True
    box = source[brow : bend + 1, bcol : cend + 1]
    black = box == 0
    output = box.copy()
    output[black] = 2

    outside = (source != 0) & ~box_mask
    blocks = _components(outside, diagonal=True)
    for cells in blocks:
        rr = [cell[0] for cell in cells]
        cc = [cell[1] for cell in cells]
        r0, c0 = min(rr), min(cc)
        patch = source[r0 : r0 + 3, c0 : c0 + 3]
        nonred = patch[(patch != 0) & (patch != 2)]
        if patch.shape != (3, 3) or not len(nonred):
            continue
        color = int(nonred[0])
        shape = patch == 2
        matches: list[tuple[int, int]] = []
        for r in range(box.shape[0] - 2):
            for c in range(box.shape[1] - 2):
                if np.array_equal(black[r : r + 3, c : c + 3], shape):
                    matches.append((r, c))
        if len(matches) != 1:
            # The generated counts are distinct; an ambiguity here is a reference
            # failure and must not be hidden by an arbitrary selection.
            raise RuntimeError(f"task233 expected one generated match, got {len(matches)}")
        r, c = matches[0]
        output[r : r + 3, c : c + 3][~shape] = color
    return output


def evaluate(task: int, cases: list[dict[str, list[list[int]]]]) -> dict[str, object]:
    right = wrong = errors = 0
    wrong_examples: list[int] = []
    for index, example in enumerate(cases):
        try:
            predicted = solve(task, example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if predicted is not None and predicted.shape == expected.shape and np.array_equal(predicted, expected):
                right += 1
            else:
                wrong += 1
                if len(wrong_examples) < 20:
                    wrong_examples.append(index)
        except Exception:  # noqa: BLE001
            errors += 1
            if len(wrong_examples) < 20:
                wrong_examples.append(index)
    total = right + wrong + errors
    return {
        "total": total,
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "rate": right / total if total else 0.0,
        "first_nonright_indices": wrong_examples,
    }


def stored(task: int) -> dict[str, object]:
    payload = json.loads((ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text())
    rows = {name: evaluate(task, payload[name]) for name in ("train", "test", "arc-gen")}
    rows["total"] = {
        key: sum(int(row[key]) for row in rows.values())
        for key in ("total", "right", "wrong", "errors")
    }
    rows["total"]["rate"] = rows["total"]["right"] / rows["total"]["total"]
    return rows


def fresh_worker(arguments: tuple[int, int, int, int]) -> dict[str, object]:
    task, start, count, seed = arguments
    random.seed(seed)
    np.random.seed(seed % (2**32))
    right = wrong = errors = 0
    wrong_examples: list[int] = []
    for offset in range(count):
        index = start + offset
        example = GENERATORS[task].generate()
        try:
            predicted = solve(task, example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if predicted is not None and predicted.shape == expected.shape and np.array_equal(predicted, expected):
                right += 1
            else:
                wrong += 1
                if len(wrong_examples) < 20:
                    wrong_examples.append(index)
        except Exception:  # noqa: BLE001
            errors += 1
            if len(wrong_examples) < 20:
                wrong_examples.append(index)
    return {
        "total": count,
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "rate": right / count,
        "first_nonright_indices": wrong_examples,
    }


def fresh(task: int, count: int, jobs: int) -> dict[str, object]:
    # Independent fixed seeds are also stronger than one long pseudo-random
    # stream and satisfy the multiple-seed requirement for private-zero review.
    jobs = max(1, min(jobs, count))
    sizes = [count // jobs + (worker < count % jobs) for worker in range(jobs)]
    starts: list[int] = []
    cursor = 0
    for size in sizes:
        starts.append(cursor)
        cursor += size
    args = [
        (task, start, size, 8_004_500 + task * 1000 + worker)
        for worker, (start, size) in enumerate(zip(starts, sizes))
    ]
    if jobs == 1:
        rows = [fresh_worker(args[0])]
    else:
        with mp.get_context("spawn").Pool(jobs) as pool:
            rows = pool.map(fresh_worker, args)
    nonright = [index for row in rows for index in row["first_nonright_indices"]]
    right = sum(int(row["right"]) for row in rows)
    wrong = sum(int(row["wrong"]) for row in rows)
    errors = sum(int(row["errors"]) for row in rows)
    return {
        "total": count,
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "rate": right / count,
        "seed_count": jobs,
        "seeds": [item[3] for item in args],
        "first_nonright_indices": sorted(nonright)[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=(18, 233, 286, 366))
    parser.add_argument("--fresh", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    count = args.fresh
    target = HERE / "reference_evidence.json"
    if target.exists():
        out: dict[str, object] = json.loads(target.read_text())
        out["fresh_requested_per_task"] = count
    else:
        out = {
        "fresh_requested_per_task": count,
        "policy_fresh_threshold": 0.90,
        "references": {},
        "task018_noninjective": {
            "deterministic_exact_mapping_possible": False,
            "reason": (
                "generator rotation can be ambiguous from the three visible marker cells; "
                "retained legal calls with byte-identical inputs and different outputs are "
                "documented in lane_c19"
            ),
            "witness_source": "scripts/golf/loop_7999_13/lane_c19/fresh_evidence.json",
        },
        }
    selected = (args.task,) if args.task is not None else (18, 233, 286, 366)
    for task in selected:
        print(f"reference task{task:03d}: stored", flush=True)
        row = {
            "source": str(Path(REFS[task].__file__).relative_to(ROOT)),
            "source_sha256": HASHES[task],
            "stored": stored(task),
        }
        print(f"reference task{task:03d}: fresh{count}", flush=True)
        row["fresh"] = fresh(task, count, args.jobs)
        row["classification"] = (
            "canonical_policy_solver_noninjective_task" if task == 18 else "exact_numpy_reference"
        )
        out["references"][str(task)] = row
        target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(task, row["stored"]["total"], row["fresh"], flush=True)


if __name__ == "__main__":
    main()
