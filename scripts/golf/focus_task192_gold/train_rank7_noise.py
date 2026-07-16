#!/usr/bin/env python3
"""Train a rank-7 spatial code only on task192's corrupted/noise cells.

The exact-ArgMax candidate can route every selected-color cell directly.  The
low-rank spatial predicate is therefore needed only at cells of the second
color, where it decides whether the pixel is a hole inside a rectangle or an
isolated pixel outside every rectangle.  This script trains that smaller
problem and writes factors/thresholds; it never touches submission authority.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as nnf
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))


def selected_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.reshape(-1), minlength=10)
    # ArgMax over channels 1..9, with ONNX select_last_index=0 tie behavior.
    return int(np.argmax(counts[1:]) + 1)


def examples_to_points(examples: list[dict[str, object]]) -> dict[str, np.ndarray]:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    inside_rows: list[np.ndarray] = []
    inside_cols: list[np.ndarray] = []
    row_indices: list[int] = []
    col_indices: list[int] = []
    labels: list[bool] = []
    horizontal_labels: list[bool] = []
    vertical_labels: list[bool] = []
    case_indices: list[int] = []
    for case_index, example in enumerate(examples):
        grid = np.asarray(example["input"], dtype=np.int8)
        expected = np.asarray(example["output"], dtype=np.int8)
        height, width = grid.shape
        color = selected_color(grid)
        selected = np.zeros((30, 30), dtype=np.float32)
        selected[:height, :width] = grid == color
        for row, col in zip(*np.where((grid != 0) & (grid != color))):
            row_vector = selected[row].copy()
            col_vector = selected[:, col].copy()
            inside_row = np.zeros(30, dtype=np.float32)
            inside_col = np.zeros(30, dtype=np.float32)
            inside_row[:width] = 1.0
            inside_col[:height] = 1.0
            rows.append(row_vector)
            cols.append(col_vector)
            inside_rows.append(inside_row)
            inside_cols.append(inside_col)
            row_indices.append(int(row))
            col_indices.append(int(col))
            labels.append(bool(expected[row, col] == color))
            horizontal_labels.append(bool(row_vector[max(0, col - 1):col + 2].any()))
            vertical_labels.append(bool(col_vector[max(0, row - 1):row + 2].any()))
            case_indices.append(case_index)
    return {
        "rows": np.asarray(rows, dtype=np.float32),
        "cols": np.asarray(cols, dtype=np.float32),
        "inside_rows": np.asarray(inside_rows, dtype=np.float32),
        "inside_cols": np.asarray(inside_cols, dtype=np.float32),
        "row_indices": np.asarray(row_indices, dtype=np.int64),
        "col_indices": np.asarray(col_indices, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.bool_),
        "horizontal_labels": np.asarray(horizontal_labels, dtype=np.bool_),
        "vertical_labels": np.asarray(vertical_labels, dtype=np.bool_),
        "case_indices": np.asarray(case_indices, dtype=np.int64),
        "case_count": np.asarray(len(examples), dtype=np.int64),
    }


def generate(count: int, seed: int) -> list[dict[str, object]]:
    generator = importlib.import_module("task_7e0986d6")
    random.seed(seed)
    result = []
    started = time.time()
    while len(result) < count:
        result.append(generator.generate())
        if len(result) % 500 == 0:
            print(json.dumps({"generated": len(result), "seconds": time.time() - started}), flush=True)
    return result


def load_or_generate(path: Path, count: int, seed: int, include_known: bool) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path) as loaded:
            data = {name: loaded[name] for name in loaded.files}
        if int(data["generated_count"]) == count and int(data["seed"]) == seed and bool(data["include_known"]) == include_known:
            return data
    examples = generate(count, seed)
    known_count = 0
    if include_known:
        known = json.loads((ROOT / "inputs/neurogolf-2026/task192.json").read_text())
        known_examples = known["train"] + known["test"] + known["arc-gen"]
        known_count = len(known_examples)
        examples = known_examples + examples
    data = examples_to_points(examples)
    data.update({
        "generated_count": np.asarray(count, dtype=np.int64),
        "known_count": np.asarray(known_count, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
        "include_known": np.asarray(include_known, dtype=np.bool_),
    })
    np.savez_compressed(path, **data)
    return data


def authority_factor() -> np.ndarray:
    with zipfile.ZipFile(ROOT / "submission_base_8014.69.zip") as archive:
        model = onnx.load_model_from_string(archive.read("task192.onnx"))
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    return arrays["eig_factor"].astype(np.float32)


def ratios(factor: torch.Tensor, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_projection = data["rows"] @ factor
    col_projection = data["cols"] @ factor
    inside_row_projection = data["inside_rows"] @ factor
    inside_col_projection = data["inside_cols"] @ factor
    at_col = factor[data["col_indices"]]
    at_row = factor[data["row_indices"]]
    horizontal = (row_projection * at_col).sum(dim=1)
    vertical = (col_projection * at_row).sum(dim=1)
    horizontal_base = (inside_row_projection * at_col).sum(dim=1)
    vertical_base = (inside_col_projection * at_row).sum(dim=1)
    base = horizontal_base * vertical_base
    ratio = horizontal * vertical / base.clamp_min(1e-8)
    return ratio, horizontal / horizontal_base.clamp_min(1e-8), vertical / vertical_base.clamp_min(1e-8)


def best_threshold(values: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    order = np.argsort(values, kind="stable")
    values = values[order]
    labels = labels[order]
    positive_total = int(labels.sum())
    positive_seen = negative_seen = 0
    best_correct = positive_total
    best = float(np.nextafter(values[0], -np.inf))
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end] == values[index]:
            end += 1
        chunk = labels[index:end]
        positive_seen += int(chunk.sum())
        negative_seen += int(len(chunk) - chunk.sum())
        correct = negative_seen + positive_total - positive_seen
        if correct > best_correct:
            best_correct = correct
            best = float(values[index])
        index = end
    return best, len(values) - best_correct


def torch_data(data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    names = ("rows", "cols", "inside_rows", "inside_cols", "row_indices", "col_indices")
    return {name: torch.from_numpy(data[name]) for name in names}


def evaluate(factor: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, object]:
    with torch.no_grad():
        ratio, horizontal, vertical = ratios(torch.from_numpy(factor), torch_data(data))
    ratio_np = ratio.numpy().astype(np.float64)
    labels = data["labels"]
    threshold, point_errors = best_threshold(ratio_np, labels)
    predicted = ratio_np > threshold
    wrong_cases = np.unique(data["case_indices"][predicted != labels])
    positives = ratio_np[labels]
    negatives = ratio_np[~labels]
    h_np, v_np = horizontal.numpy(), vertical.numpy()
    h_labels, v_labels = data["horizontal_labels"], data["vertical_labels"]
    return {
        "threshold": threshold,
        "point_errors": int(point_errors),
        "case_errors": int(len(wrong_cases)),
        "cases": int(data["case_count"]),
        "points": int(len(labels)),
        "positive_min": float(positives.min()),
        "negative_max": float(negatives.max()),
        "gap": float(positives.min() - negatives.max()),
        "base_axis_horizontal_positive_min": float(h_np[h_labels].min()),
        "base_axis_horizontal_negative_max": float(h_np[~h_labels].max()),
        "base_axis_vertical_positive_min": float(v_np[v_labels].min()),
        "base_axis_vertical_negative_max": float(v_np[~v_labels].max()),
    }


def train_one(
    initial: np.ndarray,
    train: dict[str, np.ndarray],
    steps: int,
    seed: int,
    learning_rate: float,
    axis_weight: float,
) -> tuple[np.ndarray, dict[str, object]]:
    torch.manual_seed(seed)
    factor = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    threshold = torch.nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    axis_threshold = torch.nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
    optimizer = torch.optim.Adam([factor, threshold, axis_threshold], lr=learning_rate)
    data = torch_data(train)
    labels = torch.from_numpy(train["labels"].astype(np.float32))
    h_labels = torch.from_numpy(train["horizontal_labels"].astype(np.float32))
    v_labels = torch.from_numpy(train["vertical_labels"].astype(np.float32))
    target_norm = float(np.linalg.norm(initial))
    best_factor = initial.copy()
    best = evaluate(best_factor, train)
    best["step"] = 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        ratio, horizontal, vertical = ratios(factor, data)
        signs = labels * 2.0 - 1.0
        # Log-sum-exp and mean losses together keep pressure on the single
        # hardest point without discarding the broader shape of the support.
        margins = signs * (ratio - threshold)
        point_losses = nnf.softplus(-80.0 * margins) / 80.0
        loss = point_losses.mean() + 0.08 * torch.logsumexp(40.0 * point_losses, dim=0) / 40.0
        if axis_weight:
            h_signs = h_labels * 2.0 - 1.0
            v_signs = v_labels * 2.0 - 1.0
            axis_losses = torch.cat((
                nnf.softplus(-40.0 * h_signs * (horizontal - axis_threshold)) / 40.0,
                nnf.softplus(-40.0 * v_signs * (vertical - axis_threshold)) / 40.0,
            ))
            loss = loss + axis_weight * (
                axis_losses.mean() + 0.04 * torch.logsumexp(30.0 * axis_losses, dim=0) / 30.0
            )
        # Rows 20..29 are outside every generated logical grid.  Keep them at
        # zero to avoid a numerically irrelevant degree of freedom.
        loss = loss + 0.01 * factor[20:].square().sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([factor, threshold, axis_threshold], 10.0)
        optimizer.step()
        with torch.no_grad():
            factor[20:].zero_()
            factor.mul_(target_norm / factor.norm().clamp_min(1e-8))
            threshold.clamp_(-1.0, 2.0)
            axis_threshold.clamp_(-1.0, 2.0)
        if step % 50 == 0 or step == steps:
            candidate = factor.detach().numpy().copy()
            report = evaluate(candidate, train)
            report["step"] = step
            if (report["point_errors"], -report["gap"]) < (best["point_errors"], -best["gap"]):
                best_factor, best = candidate, report
            if report["point_errors"] == 0 and report["gap"] > 1e-4:
                break
    return best_factor, best


def random_initial(current: np.ndarray, restart: int, rng: np.random.Generator) -> np.ndarray:
    if restart == 0:
        return current.copy()
    if restart < 4:
        result = current + rng.normal(0.0, 0.04 * restart, current.shape).astype(np.float32)
    else:
        result = rng.normal(0.0, 1.0, current.shape).astype(np.float32)
        result[20:] = 0
        result *= np.linalg.norm(current) / np.linalg.norm(result)
    return result


def resize_rank(current: np.ndarray, rank: int, rng: np.random.Generator) -> np.ndarray:
    if rank == current.shape[1]:
        return current
    if rank < current.shape[1]:
        return current[:, :rank].copy()
    extra = rng.normal(0.0, 0.02, (current.shape[0], rank - current.shape[1])).astype(np.float32)
    extra[20:] = 0
    return np.concatenate((current, extra), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=2_000)
    parser.add_argument("--validation-count", type=int, default=1_000)
    parser.add_argument("--train-seed", type=int, default=192_429_001)
    parser.add_argument("--validation-seed", type=int, default=192_429_002)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1_500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--axis-weight", type=float, default=0.15)
    parser.add_argument("--rank", type=int, default=7)
    args = parser.parse_args()
    train_np = load_or_generate(HERE / "rank7_noise_train.npz", args.train_count, args.train_seed, True)
    validation_np = load_or_generate(HERE / "rank7_noise_validation.npz", args.validation_count, args.validation_seed, False)
    rng = np.random.default_rng(192_429_003)
    current = resize_rank(authority_factor(), args.rank, rng)
    rows = []
    winner: tuple[np.ndarray, dict[str, object], dict[str, object]] | None = None
    for restart in range(args.restarts):
        initial = random_initial(current, restart, rng)
        factor, train_report = train_one(
            initial, train_np, args.steps, args.train_seed + restart,
            args.learning_rate, args.axis_weight,
        )
        validation_report = evaluate(factor, validation_np)
        row = {"restart": restart, "train": train_report, "validation": validation_report}
        rows.append(row)
        print(json.dumps(row), flush=True)
        key = (validation_report["point_errors"], train_report["point_errors"], -validation_report["gap"])
        if winner is None:
            winner = (factor, train_report, validation_report)
        else:
            old_key = (winner[2]["point_errors"], winner[1]["point_errors"], -winner[2]["gap"])
            if key < old_key:
                winner = (factor, train_report, validation_report)
        np.save(HERE / f"rank{args.rank}_noise_best_factor.npy", winner[0])
        (HERE / f"rank{args.rank}_noise_training.json").write_text(json.dumps({
            "args": vars(args), "runs": rows,
            "best": {"train": winner[1], "validation": winner[2]},
        }, indent=2) + "\n")
        if winner[1]["point_errors"] == 0 and winner[2]["point_errors"] == 0:
            break


if __name__ == "__main__":
    main()
