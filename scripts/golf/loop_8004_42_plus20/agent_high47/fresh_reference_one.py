#!/usr/bin/env python3
"""Generator-reference and dual-ORT fresh audit for one high47 incumbent."""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS))
from lib import scoring  # noqa: E402


HASHES = {
    44: "228f6490",
    12: "0962bcdd",
    198: "83302e8f",
    277: "b230c067",
    117: "4c5c2cf0",
    270: "ae3edfdc",
    19: "10fcaaa3",
    62: "2bcee788",
}
RULES = {
    44: "Match each outside colored creature to the equal black hole pattern inside a gray box; erase the outside copy and fill the matching hole with its color, retaining dust.",
    12: "For each two-color plus, extend its cardinal arms to radius two and add the center-color diagonals at radii one and two, preserving the gravity rotation.",
    198: "Treat colored periodic lines as cell borders; every black hole in a border becomes yellow and makes both adjacent large cells yellow, while untouched cells become green.",
    277: "Three cyan placements contain two translated copies of the full sprite and one column-deleted sprite; recolor the full copies blue and the smaller copy red.",
    117: "Find the five-cell X center and reflect the differently colored leg sprite across both center axes, producing four symmetric copies.",
    270: "Keep the two flower centers and move each same-color loose cardinal petal to the immediately adjacent cell on the same ray.",
    19: "Tile the input two by two and paint cyan on every in-bounds diagonal neighbor of each repeated nonzero source, then repaint the sources over cyan.",
    62: "Use the red seam to reflect the colored 3x3 sprite across that seam and emit the completed sprite on a green 10x10 canvas.",
}


def normalize(value):
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def raw_solver(task: int) -> Callable[[list[list[int]]], list[list[int]]]:
    path = ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"high47_raw_{task:03d}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def solve(grid: list[list[int]]) -> list[list[int]]:
        return normalize(module.p(copy.deepcopy(grid)))

    return solve


def solve_012(grid: list[list[int]]) -> list[list[int]]:
    a = np.asarray(grid, dtype=np.int8)
    colors, counts = np.unique(a[a != 0], return_counts=True)
    if len(colors) != 2:
        raise ValueError("expected two nonzero colors")
    center_color = int(colors[int(np.argmin(counts))])
    arm_color = int(colors[int(np.argmax(counts))])
    out = a.copy()
    for r, c in np.argwhere(a == center_color):
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1), (-2, -2), (-2, 2), (2, -2), (2, 2)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]:
                out[rr, cc] = center_color
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]:
                out[rr, cc] = arm_color
    return out.tolist()


def solve_019(grid: list[list[int]]) -> list[list[int]]:
    a = np.asarray(grid, dtype=np.int8)
    h, w = a.shape
    out = np.tile(a, (2, 2))
    sources = np.argwhere(out != 0)
    for r, c in sources:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < 2 * h and 0 <= cc < 2 * w and out[rr, cc] == 0:
                out[rr, cc] = 8
    return out.tolist()


def solve_117(grid: list[list[int]]) -> list[list[int]]:
    a = np.asarray(grid, dtype=np.int8)
    h, w = a.shape
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(a[a != 0]):
        points = {tuple(map(int, item)) for item in np.argwhere(a == color)}
        if len(points) != 5:
            continue
        for r, c in points:
            x = {(r, c), (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)}
            if points == x:
                other = np.argwhere((a != 0) & (a != int(color)))
                one_quadrant = other.size and all(
                    (int(rr) - r) * (int(other[0, 0]) - r) > 0
                    and (int(cc) - c) * (int(other[0, 1]) - c) > 0
                    for rr, cc in other
                )
                all_reflections_in_bounds = all(
                    0 <= rr < h and 0 <= c2 < w
                    for rr0, cc0 in other
                    for rr in (int(rr0), 2 * r - int(rr0))
                    for c2 in (int(cc0), 2 * c - int(cc0))
                )
                if one_quadrant and all_reflections_in_bounds:
                    candidates.append((r, c, int(color)))
    if len(candidates) != 1:
        raise ValueError(f"X center candidates={candidates}")
    cr, cc, body = candidates[0]
    out = a.copy()
    for r, c in np.argwhere((a != 0) & (a != body)):
        color = int(a[r, c])
        for rr in (int(r), 2 * cr - int(r)):
            for c2 in (int(c), 2 * cc - int(c)):
                if 0 <= rr < h and 0 <= c2 < w:
                    out[rr, c2] = color
    return out.tolist()


def solve_198(grid: list[list[int]]) -> list[list[int]]:
    a = np.asarray(grid, dtype=np.int8)
    h, w = a.shape
    colors, counts = np.unique(a[a != 0], return_counts=True)
    if not len(colors):
        raise ValueError("missing line color")
    line_color = int(colors[int(np.argmax(counts))])
    row_scores = np.count_nonzero(a == line_color, axis=1)
    col_scores = np.count_nonzero(a == line_color, axis=0)
    line_rows = [int(i) for i in np.flatnonzero(row_scores >= max(2, w // 2))]
    line_cols = [int(i) for i in np.flatnonzero(col_scores >= max(2, h // 2))]
    if len(line_rows) < 2 or len(line_cols) < 2:
        raise ValueError("cannot infer periodic line grid")
    out = np.full_like(a, 3)
    out[line_rows, :] = line_color
    out[:, line_cols] = line_color
    holes = np.argwhere(((np.isin(np.arange(h)[:, None], line_rows)) | (np.isin(np.arange(w)[None, :], line_cols))) & (a == 0))
    for r, c in holes:
        r, c = int(r), int(c)
        out[r, c] = 4
        if r in line_rows and c not in line_cols:
            ri = line_rows.index(r)
            left = max(x for x in [-1, *line_cols] if x < c)
            right = min(x for x in [*line_cols, w] if x > c)
            above = (line_rows[ri - 1] if ri > 0 else -1, r)
            below = (r, line_rows[ri + 1] if ri + 1 < len(line_rows) else h)
            for r0, r1 in (above, below):
                if r1 > r0:
                    out[r0 + 1 : r1, left + 1 : right] = 4
        if c in line_cols and r not in line_rows:
            ci = line_cols.index(c)
            top = max(x for x in [-1, *line_rows] if x < r)
            bottom = min(x for x in [*line_rows, h] if x > r)
            left_band = (line_cols[ci - 1] if ci > 0 else -1, c)
            right_band = (c, line_cols[ci + 1] if ci + 1 < len(line_cols) else w)
            for c0, c1 in (left_band, right_band):
                if c1 > c0:
                    out[top + 1 : bottom, c0 + 1 : c1] = 4
    return out.tolist()


def solve_270(grid: list[list[int]]) -> list[list[int]]:
    a = np.asarray(grid, dtype=np.int8)
    out = np.zeros_like(a)
    for center_color, petal_color in ((2, 3), (1, 7)):
        centers = np.argwhere(a == center_color)
        if len(centers) != 1:
            raise ValueError(f"center {center_color} count={len(centers)}")
        cr, cc = map(int, centers[0])
        out[cr, cc] = center_color
        for r, c in np.argwhere(a == petal_color):
            r, c = int(r), int(c)
            if r == cr and c != cc:
                out[cr, cc + (1 if c > cc else -1)] = petal_color
            elif c == cc and r != cr:
                out[cr + (1 if r > cr else -1), cc] = petal_color
            else:
                raise ValueError("petal not cardinal with center")
    return out.tolist()


def solve_277(grid: list[list[int]]) -> list[list[int]]:
    """Identify the two repeated full connected components exactly.

    The generator's full sprite is 4-connected and placed twice.  Deleting a
    column can split the smaller sprite, so it can contribute one or two
    components.  The repeated largest normalized component is therefore the
    full sprite; every remaining cyan component belongs to the reduced copy.
    """
    a = np.asarray(grid, dtype=np.int8)
    unseen = {tuple(map(int, item)) for item in np.argwhere(a == 8)}
    components: list[set[tuple[int, int]]] = []
    while unseen:
        start = unseen.pop()
        component = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    point = (r + dr, c + dc)
                    if point in unseen:
                        unseen.remove(point)
                        component.add(point)
                        stack.append(point)
        components.append(component)

    def normalized(component: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
        r0 = min(r for r, _ in component)
        c0 = min(c for _, c in component)
        return tuple(sorted((r - r0, c - c0) for r, c in component))

    groups: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for index, component in enumerate(components):
        groups.setdefault(normalized(component), []).append(index)
    repeated = [indices for indices in groups.values() if len(indices) == 2]
    if not repeated:
        raise ValueError("no repeated full component")
    largest = max(len(components[indices[0]]) for indices in repeated)
    repeated = [indices for indices in repeated if len(components[indices[0]]) == largest]
    if len(repeated) != 1:
        raise ValueError("repeated full component is ambiguous")
    blue = components[repeated[0][0]] | components[repeated[0][1]]
    out = np.zeros_like(a)
    for component in components:
        for point in component:
            out[point] = 1 if point in blue else 2
    return out.tolist()


SOLVERS: dict[int, Callable[[list[list[int]]], list[list[int]]]] = {
    12: solve_012,
    19: solve_019,
    117: solve_117,
    198: solve_198,
    270: solve_270,
    277: solve_277,
}


def make_session(model: onnx.ModelProto, disable: bool):
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def model_one(session, example: dict) -> tuple[bool, int]:
    benchmark = scoring.convert_to_numpy(example)
    if benchmark is None:
        raise ValueError("example conversion failed")
    raw = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: benchmark["input"]})[0]
    return bool(np.array_equal(raw > 0, benchmark["output"] > 0)), int(np.count_nonzero((raw > 0) & (raw < 0.25)))


def reference_batch(solver, examples: list[dict]) -> dict[str, object]:
    right = wrong = errors = 0
    first_failure = None
    for index, example in enumerate(examples):
        try:
            actual = normalize(solver(copy.deepcopy(example["input"])))
            if actual == example["output"]:
                right += 1
            else:
                wrong += 1
                if first_failure is None:
                    first_failure = {"index": index, "kind": "wrong"}
        except Exception as exc:
            errors += 1
            if first_failure is None:
                first_failure = {"index": index, "kind": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {"right": right, "wrong": wrong, "errors": errors, "total": len(examples), "first_failure": first_failure}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=sorted(HASHES), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ort.set_default_logger_severity(4)

    task = args.task
    solver = SOLVERS.get(task) or raw_solver(task)
    known_data = scoring.load_examples(task)
    known = [example for split in ("train", "test", "arc-gen") for example in known_data[split]]
    known_reference = reference_batch(solver, known)
    generator = importlib.import_module(f"task_{HASHES[task]}")
    model = onnx.load(args.model)
    sessions: dict[str, object] = {}
    for name, disable in (("disable_all", True), ("default", False)):
        try:
            sessions[name] = make_session(model, disable)
        except Exception as exc:
            sessions[name] = f"{type(exc).__name__}: {exc}"

    seeds = (47_000_001 + task, 47_100_001 + task)
    fresh_rows = []
    for seed in seeds:
        random.seed(seed)
        examples = [generator.generate() for _ in range(args.count)]
        ref = reference_batch(solver, examples)
        modes: dict[str, object] = {}
        decoded: dict[str, list[bool]] = {}
        for name, session in sessions.items():
            if isinstance(session, str):
                modes[name] = {"session_error": session, "right": 0, "wrong": 0, "errors": args.count}
                continue
            right = wrong = errors = near = 0
            results: list[bool] = []
            for example in examples:
                try:
                    ok, unsafe = model_one(session, example)
                    results.append(ok)
                    near += unsafe
                    right += int(ok)
                    wrong += int(not ok)
                except Exception:
                    results.append(False)
                    errors += 1
            decoded[name] = results
            modes[name] = {
                "right": right,
                "wrong": wrong,
                "errors": errors,
                "total": args.count,
                "rate": right / args.count,
                "near_margin_count": near,
            }
        disagreements = None
        if set(decoded) == {"disable_all", "default"}:
            disagreements = sum(a != b for a, b in zip(decoded["disable_all"], decoded["default"]))
        fresh_rows.append({"seed": seed, "reference": ref, "model": modes, "mode_result_disagreements": disagreements})

    result = {
        "task": task,
        "generator_hash": HASHES[task],
        "rule_summary": RULES[task],
        "reference_implementation": "readable high47 solver" if task in SOLVERS else "independent Sakana input-only solver",
        "known_reference": known_reference,
        "fresh_count_each": args.count,
        "fresh": fresh_rows,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"task{task:03d} ref_known={known_reference['right']}/{known_reference['total']} "
        + " ".join(
            f"seed={row['seed']} ref={row['reference']['right']}/{args.count} "
            f"off={row['model'].get('disable_all', {}).get('right', 0)}/{args.count} "
            f"def={row['model'].get('default', {}).get('right', 0)}/{args.count}"
            for row in fresh_rows
        )
    )


if __name__ == "__main__":
    main()
