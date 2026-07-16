#!/usr/bin/env python3
"""Independent strict audit for the target-mid17 NeuroGolf lane.

This script deliberately does not promote or mutate shared artifacts.  It audits
the four exact members extracted from submission_base_8004.50.zip, the only
historical sub-baseline leads, and independently checks decoded generator rules.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))

from lib import scoring  # noqa: E402


HASHES = {
    184: "780d0b14",
    268: "aba27056",
    324: "d07ae81c",
    338: "d5d6de2d",
}
FRESH_SEEDS = (8_004_421_701, 8_004_421_702)


def load_shared_auditor():
    path = ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py"
    spec = importlib.util.spec_from_file_location("target_mid17_shared_auditor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solve_324(grid: list[list[int]]) -> list[list[int]]:
    """Decode d07ae81c from input: extend both diagonals through every seed."""
    a = np.asarray(grid, dtype=np.int8)
    h, w = a.shape
    counts = np.bincount(a.reshape(-1), minlength=10)
    corner_colors = set(int(x) for x in a[:2, :2].reshape(-1))
    bg0 = max(corner_colors, key=lambda color: int(counts[color]))
    bg1 = max(
        (color for color in range(1, 10) if color != bg0),
        key=lambda color: int(counts[color]),
    )
    seeds = np.argwhere((a != bg0) & (a != bg1))
    row_is_stripe = ~np.any(a == bg0, axis=1)
    col_is_stripe = ~np.any(a == bg0, axis=0)
    paint: dict[bool, int] = {}
    for r, c in seeds:
        paint[bool(row_is_stripe[r] or col_is_stripe[c])] = int(a[r, c])
    if set(paint) != {False, True}:
        raise ValueError("input does not expose both generator seed colors")
    diag_sum = set(int(r + c) for r, c in seeds)
    diag_diff = set(int(r - c) for r, c in seeds)
    out = a.copy()
    for r in range(h):
        for c in range(w):
            if r + c not in diag_sum and r - c not in diag_diff:
                continue
            if a[r, c] == bg0:
                out[r, c] = paint[False]
            elif a[r, c] == bg1:
                out[r, c] = paint[True]
    return out.tolist()


def solve_338(grid: list[list[int]]) -> list[list[int]]:
    """Decode d5d6de2d: green is the black region enclosed by red walls."""
    a = np.asarray(grid, dtype=np.int8)
    h, w = a.shape
    reachable = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if a[r, c] != 2 and not reachable[r, c]:
                reachable[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if a[r, c] != 2 and not reachable[r, c]:
                reachable[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and a[rr, cc] != 2 and not reachable[rr, cc]:
                reachable[rr, cc] = True
                queue.append((rr, cc))
    out = np.zeros_like(a)
    out[(a == 0) & ~reachable] = 3
    return out.tolist()


def solve_268(grid: list[list[int]]) -> list[list[int]]:
    """Decode aba27056 by canonicalizing the unique open side of the box."""
    a = np.asarray(grid, dtype=np.int8)
    h, w = a.shape
    nz = a != 0
    rows = np.flatnonzero(np.any(nz, axis=1))
    cols = np.flatnonzero(np.any(nz, axis=0))
    if rows.size == 0 or cols.size == 0:
        return a.tolist()
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(cols[0]), int(cols[-1])
    edge_zeros = (
        int(np.count_nonzero(a[top, left : right + 1] == 0)),
        int(np.count_nonzero(a[bottom, left : right + 1] == 0)),
        int(np.count_nonzero(a[top : bottom + 1, left] == 0)),
        int(np.count_nonzero(a[top : bottom + 1, right] == 0)),
    )
    rr, cc = np.indices(a.shape)
    if edge_zeros[0] > edge_zeros[1]:
        rp, cp, tp, bt, lf, rt = rr, cc, top, bottom, left, right
    elif edge_zeros[1] > edge_zeros[0]:
        rp, cp = h - 1 - rr, cc
        tp, bt, lf, rt = h - 1 - bottom, h - 1 - top, left, right
    elif edge_zeros[2] > edge_zeros[3]:
        rp, cp, tp, bt, lf, rt = cc, rr, left, right, top, bottom
    elif edge_zeros[3] > edge_zeros[2]:
        rp, cp = w - 1 - cc, rr
        tp, bt, lf, rt = w - 1 - right, w - 1 - left, top, bottom
    else:
        raise ValueError("no unique open box side")
    inside = (tp < rp) & (rp < bt) & (lf < cp) & (cp < rt)
    column = (rp < bt) & (lf + 2 <= cp) & (cp <= rt - 2)
    rays = (rp < tp) & (
        ((cp - rp) == (lf + 2 - tp)) | ((cp + rp) == (rt - 2 + tp))
    )
    out = a.copy()
    out[inside | column | rays] = 4
    return out.tolist()


def _bands(present: list[bool]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(present + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            bands.append((start, index))
            start = None
    return bands


def solve_184(grid: list[list[int]]) -> list[list[int]]:
    """Decode recoverable 780d0b14 instances using zero separator bands."""
    h, w = len(grid), len(grid[0])
    row_bands = _bands([any(grid[r][c] != 0 for c in range(w)) for r in range(h)])
    col_bands = _bands([any(grid[r][c] != 0 for r in range(h)) for c in range(w)])
    out: list[list[int]] = []
    for r0, r1 in row_bands:
        row: list[int] = []
        for c0, c1 in col_bands:
            colors = {
                grid[r][c]
                for r in range(r0, r1)
                for c in range(c0, c1)
                if grid[r][c] != 0
            }
            if len(colors) != 1:
                raise ValueError("erasure made a patch ambiguous")
            row.append(next(iter(colors)))
        out.append(row)
    return out


SOLVERS: dict[int, Callable[[list[list[int]]], list[list[int]]]] = {
    184: solve_184,
    268: solve_268,
    324: solve_324,
    338: solve_338,
}


def reference_batch(task: int, examples: list[dict]) -> dict[str, int]:
    right = wrong = undecodable = compatible = 0
    for example in examples:
        compatible += int(max(len(example["input"]), len(example["input"][0])) <= 30)
        try:
            actual = SOLVERS[task](example["input"])
        except ValueError:
            undecodable += 1
            continue
        if actual == example["output"]:
            right += 1
        else:
            wrong += 1
    return {
        "total": len(examples),
        "right": right,
        "wrong": wrong,
        "undecodable": undecodable,
        "onnx_30x30_compatible": compatible,
    }


def generate_324_terminating(generator):
    """Sample d07ae81c conditional on the official generator returning.

    The upstream implementation can enter an infinite loop when both stripe
    lists are empty because it then asks dots to observe two background colors.
    Such a call produces no example.  Rejecting that parameter draw is therefore
    the appropriate distribution over returned examples and keeps a 5k audit
    finite.
    """
    common = importlib.import_module("common")
    while True:
        width, height = common.randint(10, 20), common.randint(10, 20)
        bgcolors = common.random_colors(2)
        colors = common.random_colors(2, exclude=bgcolors)
        brows: list[int] = []
        bcols: list[int] = []
        row, col = common.randint(2, 7), common.randint(2, 7)
        while True:
            spacing = common.randint(2, 5)
            if row + spacing > height:
                break
            brows.extend(range(row, row + spacing))
            row += spacing + common.randint(4, 5)
        while True:
            spacing = common.randint(2, 5)
            if col + spacing > width:
                break
            bcols.extend(range(col, col + spacing))
            col += spacing + common.randint(4, 5)
        if not brows and not bcols:
            continue
        while True:
            pixels = common.sample(common.all_pixels(width, height), common.randint(2, 3))
            pixels = common.remove_diagonal_neighbors(pixels)
            seen = {bgcolors[1] if r in brows or c in bcols else bgcolors[0] for r, c in pixels}
            if len(seen) == 2:
                break
        rows, cols = zip(*pixels)
        return generator.generate(
            width=width,
            height=height,
            rows=rows,
            cols=cols,
            bgcolors=bgcolors,
            colors=colors,
            brows=brows,
            bcols=bcols,
        )


def reference_generated(task: int, generator, seed: int, count: int) -> dict[str, int]:
    random.seed(seed)
    right = wrong = undecodable = compatible = 0
    for _ in range(count):
        example = generate_324_terminating(generator) if task == 324 else generator.generate()
        compatible += int(max(len(example["input"]), len(example["input"][0])) <= 30)
        try:
            actual = SOLVERS[task](example["input"])
        except ValueError:
            undecodable += 1
            continue
        if actual == example["output"]:
            right += 1
        else:
            wrong += 1
    return {
        "total": count,
        "right": right,
        "wrong": wrong,
        "undecodable": undecodable,
        "onnx_30x30_compatible": compatible,
    }


def reference_audit() -> dict[str, object]:
    output: dict[str, object] = {}
    for task, hash_ in HASHES.items():
        data = scoring.load_examples(task)
        known = [example for subset in ("train", "test", "arc-gen") for example in data[subset]]
        generator = importlib.import_module(f"task_{hash_}")
        fresh: dict[str, object] = {}
        for seed in FRESH_SEEDS:
            fresh[str(seed)] = reference_generated(task, generator, seed, 5000)
        output[f"task{task:03d}"] = {
            "generator_hash": hash_,
            "known": reference_batch(task, known),
            "fresh": fresh,
        }
    return output


def raw_session(path: Path) -> ort.InferenceSession:
    model = onnx.load(path)
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), providers=["CPUExecutionProvider"], sess_options=options
    )


def onnx_fresh_268(path: Path) -> dict[str, object]:
    """Two independent 5k runs for the only cheaper complete-known lead."""
    session = raw_session(path)
    generator = importlib.import_module("task_aba27056")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output: dict[str, object] = {}
    for seed in FRESH_SEEDS:
        random.seed(seed)
        right = wrong = errors = 0
        first_mismatch: int | None = None
        for index in range(5000):
            example = generator.generate()
            benchmark = scoring.convert_to_numpy(example)
            assert benchmark is not None
            try:
                actual = session.run([output_name], {input_name: benchmark["input"]})[0] > 0
                if np.array_equal(actual, benchmark["output"] > 0):
                    right += 1
                else:
                    wrong += 1
                    if first_mismatch is None:
                        first_mismatch = index
            except Exception:  # noqa: BLE001
                errors += 1
        output[str(seed)] = {
            "total": 5000,
            "right": right,
            "wrong": wrong,
            "errors": errors,
            "rate": right / 5000,
            "first_mismatch": first_mismatch,
        }
    return output


def main() -> None:
    ort.set_default_logger_severity(4)
    auditor = load_shared_auditor()
    models = {
        "base_task324": (324, HERE / "baseline/task324.onnx"),
        "base_task338": (338, HERE / "baseline/task338.onnx"),
        "base_task268": (268, HERE / "baseline/task268.onnx"),
        "base_task184": (184, HERE / "baseline/task184.onnx"),
        "history_task338_cost424": (
            338,
            ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task338_r02_static424.onnx",
        ),
        "history_task268_cost327": (
            268,
            ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task268_r01_static327.onnx",
        ),
        "history_task184_cost422": (
            184,
            ROOT / "scripts/golf/loop_7999_13/lane_c27/candidates/task184_sound422_r06.onnx",
        ),
    }
    controls = {
        "control_task324_rule16550": (
            324,
            ROOT / "scripts/golf/scratch/task324/cand_b_count_only.onnx",
        ),
        "control_task338_giant_einsum740": (
            338,
            ROOT / "scripts/golf/scratch_codex/task338/final_cost740.onnx",
        ),
        "control_task338_rule37101": (
            338,
            ROOT / "artifacts/optimized/task338.onnx",
        ),
        "control_task338_anchor_exposed": (
            338,
            ROOT / "scripts/golf/loop_7999_13/lane_a5/candidates/task338_cast_attr.onnx",
        ),
        "control_task268_rule18665": (
            268,
            ROOT / "artifacts/optimized/task268.onnx",
        ),
        "control_task184_rule1996": (
            184,
            ROOT / "scripts/golf/scratch_codex/task184/task184_exact_vector_masks.onnx",
        ),
    }
    structural_path = HERE / "STRUCTURAL_AUDIT.json"
    if structural_path.exists():
        structural = json.loads(structural_path.read_text(encoding="utf-8"))
    else:
        structural = {}
        for label, (task, path) in models.items():
            structural[label] = auditor.audit(label, task, path)
            print(label, flush=True)
        structural_path.write_text(json.dumps(structural, indent=2) + "\n", encoding="utf-8")
    control_path = HERE / "CONTROL_AUDIT.json"
    control_output = (
        json.loads(control_path.read_text(encoding="utf-8")) if control_path.exists() else {}
    )
    missing_controls = [label for label in controls if label not in control_output]
    if missing_controls:
        for label in missing_controls:
            task, path = controls[label]
            control_output[label] = auditor.audit(label, task, path)
            print(label, flush=True)
        control_path.write_text(json.dumps(control_output, indent=2) + "\n", encoding="utf-8")
    reference = reference_audit()
    (HERE / "REFERENCE_AUDIT.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    candidate_fresh = onnx_fresh_268(models["history_task268_cost327"][1])
    (HERE / "CANDIDATE_FRESH.json").write_text(
        json.dumps(
            {
                "task": 268,
                "candidate": str(models["history_task268_cost327"][1].relative_to(ROOT)),
                "lineage": "historical private/lookup reconstruction",
                "required_rate": 1.0,
                "runs": candidate_fresh,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
