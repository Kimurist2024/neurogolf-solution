#!/usr/bin/env python3
"""Independent NumPy reference for ARC-GEN task 6aa20dc0 (task158).

The generator exposes one complete 3x3 diagonally-symmetric sprite and then
places two to four non-overlapping magnified/flipped copies.  In every copy
after the first, only the two opposite-corner endpoint colours remain visible;
the other sprite cells must be restored from the complete source sprite.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
TASK_JSON = ROOT / "inputs/neurogolf-2026/task158.json"


def mode_colour(grid: np.ndarray) -> int:
    return int(np.argmax(np.bincount(grid.reshape(-1), minlength=10)))


def find_complete_sprite(grid: np.ndarray, background: int) -> np.ndarray:
    """Find the only 3x3 window containing all three sprite colours."""
    height, width = grid.shape
    matches: list[tuple[int, int, int, int]] = []
    for row in range(height - 2):
        for col in range(width - 2):
            patch = grid[row : row + 3, col : col + 3]
            occupied = patch[patch != background]
            if occupied.size < 5:
                continue
            distinct = len(np.unique(occupied))
            if distinct < 3:
                continue
            matches.append((distinct, int(occupied.size), row, col))
    if not matches:
        raise ValueError("complete 3x3 source sprite not found")
    _, _, row, col = max(matches)
    return grid[row : row + 3, col : col + 3].copy()


def endpoint_cells(source: np.ndarray, background: int) -> list[tuple[int, int, int]]:
    """Return the two unique-colour opposite corners of the source sprite."""
    result: list[tuple[int, int, int]] = []
    for row, col in ((0, 0), (0, 2), (2, 0), (2, 2)):
        colour = int(source[row, col])
        if colour != background and int(np.count_nonzero(source == colour)) == 1:
            result.append((row, col, colour))
    if len(result) != 2:
        raise ValueError(f"expected two endpoint cells, got {result}")
    first, second = result
    if first[0] + second[0] != 2 or first[1] + second[1] != 2:
        raise ValueError(f"endpoint cells are not opposite corners: {result}")
    return result


def is_maximal_colour_block(
    grid: np.ndarray,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    colour: int,
) -> bool:
    """Reject a sub-block inside a larger magnified endpoint block."""
    height, width = grid.shape
    if row0 and np.any(grid[row0 - 1, col0:col1] == colour):
        return False
    if row1 < height and np.any(grid[row1, col0:col1] == colour):
        return False
    if col0 and np.any(grid[row0:row1, col0 - 1] == colour):
        return False
    if col1 < width and np.any(grid[row0:row1, col1] == colour):
        return False
    return True


def solve_grid(input_grid: list[list[int]] | np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=np.int64)
    height, width = grid.shape
    background = mode_colour(grid)
    source = find_complete_sprite(grid, background)
    endpoints = endpoint_cells(source, background)
    endpoint_positions = {(row, col) for row, col, _ in endpoints}
    hidden_cells = [
        (row, col, int(source[row, col]))
        for row in range(3)
        for col in range(3)
        if source[row, col] != background and (row, col) not in endpoint_positions
    ]

    # Enumerate the exact generator family: magnification 1..3 and both flips.
    candidates: list[
        tuple[int, int, int, int, int, tuple[tuple[int, int, int, int], ...]]
    ] = []
    for magnitude in (1, 2, 3):
        size = 3 * magnitude
        for top in range(height - size + 1):
            for left in range(width - size + 1):
                for vertical_flip in (0, 1):
                    for horizontal_flip in (0, 1):
                        visible = np.zeros((size, size), dtype=bool)
                        endpoint_keys: list[tuple[int, int, int, int]] = []
                        valid = True
                        for source_row, source_col, colour in endpoints:
                            target_row = 2 - source_row if vertical_flip else source_row
                            target_col = 2 - source_col if horizontal_flip else source_col
                            row0 = top + target_row * magnitude
                            col0 = left + target_col * magnitude
                            block = grid[
                                row0 : row0 + magnitude,
                                col0 : col0 + magnitude,
                            ]
                            if block.shape != (magnitude, magnitude) or not np.all(block == colour):
                                valid = False
                                break
                            if not is_maximal_colour_block(
                                grid,
                                row0,
                                row0 + magnitude,
                                col0,
                                col0 + magnitude,
                                colour,
                            ):
                                valid = False
                                break
                            visible[
                                target_row * magnitude : (target_row + 1) * magnitude,
                                target_col * magnitude : (target_col + 1) * magnitude,
                            ] = True
                            endpoint_keys.append((colour, row0, col0, magnitude))
                        if not valid:
                            continue
                        box = grid[top : top + size, left : left + size]
                        if np.all(box[~visible] == background):
                            candidates.append(
                                (
                                    top,
                                    left,
                                    size,
                                    vertical_flip,
                                    horizontal_flip,
                                    tuple(sorted(endpoint_keys)),
                                )
                            )

    # A local endpoint pair can occasionally form a false diagonal with an
    # endpoint from another object.  Recover the generator's global placement
    # set instead: every maximal endpoint block must be used exactly once and
    # selected 3m boxes must obey its two-cell spacing constraint.
    universe = {key for candidate in candidates for key in candidate[5]}
    selected: list[
        tuple[int, int, int, int, int, tuple[tuple[int, int, int, int], ...]]
    ] = []

    def spaced(
        candidate: tuple[int, int, int, int, int, tuple[tuple[int, int, int, int], ...]],
        chosen: list[
            tuple[int, int, int, int, int, tuple[tuple[int, int, int, int], ...]]
        ],
    ) -> bool:
        top, left, size = candidate[:3]
        return all(
            top + size + 2 <= other_top
            or other_top + other_size + 2 <= top
            or left + size + 2 <= other_left
            or other_left + other_size + 2 <= left
            for other_top, other_left, other_size, *_ in chosen
        )

    def exact_cover(
        used: set[tuple[int, int, int, int]],
        chosen: list[
            tuple[int, int, int, int, int, tuple[tuple[int, int, int, int], ...]]
        ],
    ) -> bool:
        if used == universe:
            selected.extend(chosen)
            return True
        next_key = min(universe - used)
        for candidate in sorted(candidates):
            keys = set(candidate[5])
            if next_key not in keys or keys & used or not spaced(candidate, chosen):
                continue
            if exact_cover(used | keys, [*chosen, candidate]):
                return True
        return False

    if universe and not exact_cover(set(), []):
        raise ValueError("endpoint components have no generator-valid exact cover")

    result = grid.copy()
    for top, left, size, vertical_flip, horizontal_flip, _ in selected:
        magnitude = size // 3
        for source_row, source_col, colour in hidden_cells:
            target_row = 2 - source_row if vertical_flip else source_row
            target_col = 2 - source_col if horizontal_flip else source_col
            result[
                top + target_row * magnitude : top + (target_row + 1) * magnitude,
                left + target_col * magnitude : left + (target_col + 1) * magnitude,
            ] = colour
    return result


def verify_known() -> dict[str, object]:
    data = json.loads(TASK_JSON.read_text())
    subsets: dict[str, dict[str, int]] = {}
    first_failure = None
    for subset in ("train", "test", "arc-gen"):
        right = wrong = 0
        for index, example in enumerate(data[subset]):
            actual = solve_grid(example["input"])
            expected = np.asarray(example["output"], dtype=np.int64)
            if np.array_equal(actual, expected):
                right += 1
            else:
                wrong += 1
                if first_failure is None:
                    first_failure = {
                        "subset": subset,
                        "index": index,
                        "different_cells": int(np.count_nonzero(actual != expected)),
                    }
        subsets[subset] = {"right": right, "wrong": wrong}
    return {
        "subsets": subsets,
        "right": sum(row["right"] for row in subsets.values()),
        "wrong": sum(row["wrong"] for row in subsets.values()),
        "first_failure": first_failure,
    }


def verify_fresh(seed: int, count: int) -> dict[str, object]:
    sys.path.insert(0, str(TASKS_DIR))
    generator = importlib.import_module("task_6aa20dc0")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    right = wrong = generation_errors = 0
    first_failure = None
    shapes: Counter[str] = Counter()
    magnitudes: Counter[str] = Counter()
    mega_counts: Counter[str] = Counter()
    for index in range(count):
        try:
            example = generator.generate()
        except Exception as exc:  # noqa: BLE001
            generation_errors += 1
            if first_failure is None:
                first_failure = {
                    "index": index,
                    "stage": "generation",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            continue
        expected = np.asarray(example["output"], dtype=np.int64)
        actual = solve_grid(example["input"])
        shapes[f"{expected.shape[0]}x{expected.shape[1]}"] += 1
        if np.array_equal(actual, expected):
            right += 1
        else:
            wrong += 1
            if first_failure is None:
                first_failure = {
                    "index": index,
                    "stage": "reference",
                    "different_cells": int(np.count_nonzero(actual != expected)),
                }
    return {
        "seed": seed,
        "requested": count,
        "right": right,
        "wrong": wrong,
        "generation_errors": generation_errors,
        "first_failure": first_failure,
        "shapes": dict(sorted(shapes.items())),
        "passed": right == count and wrong == 0 and generation_errors == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[158_046_1, 158_046_2])
    parser.add_argument("--output", type=Path, default=HERE / "evidence/reference_validation.json")
    args = parser.parse_args()
    result = {
        "task": 158,
        "task_hash": "6aa20dc0",
        "rule": (
            "recover the complete 3x3 symmetric source sprite; detect every "
            "magnified/flipped endpoint-only copy; restore its hidden cells"
        ),
        "known": verify_known(),
        "fresh": [verify_fresh(seed, args.count) for seed in args.seeds],
    }
    result["passed"] = (
        result["known"]["wrong"] == 0
        and all(row["passed"] for row in result["fresh"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
