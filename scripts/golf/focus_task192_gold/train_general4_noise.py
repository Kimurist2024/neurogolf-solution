#!/usr/bin/env python3
"""Train a general rank-4 (left/right) task192 spatial kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nnf

import train_rank7_noise as common


HERE = Path(__file__).resolve().parent


def ratios(
    left: torch.Tensor, right: torch.Tensor, data: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # K = left @ right.T.  Horizontal uses row-vector @ K while vertical
    # uses K @ column-vector, hence the intentionally opposite projections.
    horizontal = ((data["rows"] @ left) * right[data["col_indices"]]).sum(dim=1)
    horizontal_base = (
        (data["inside_rows"] @ left) * right[data["col_indices"]]
    ).sum(dim=1)
    vertical = ((data["cols"] @ right) * left[data["row_indices"]]).sum(dim=1)
    vertical_base = (
        (data["inside_cols"] @ right) * left[data["row_indices"]]
    ).sum(dim=1)
    base = horizontal_base * vertical_base
    ratio = horizontal * vertical / base.clamp_min(1e-8)
    return ratio, horizontal / horizontal_base.clamp_min(1e-8), vertical / vertical_base.clamp_min(1e-8), base


def evaluate(left: np.ndarray, right: np.ndarray, source: dict[str, np.ndarray]) -> dict[str, object]:
    with torch.no_grad():
        ratio, horizontal, vertical, base = ratios(
            torch.from_numpy(left), torch.from_numpy(right), common.torch_data(source)
        )
    values = ratio.numpy().astype(np.float64)
    labels = source["labels"]
    threshold, point_errors = common.best_threshold(values, labels)
    predicted = values > threshold
    bad = predicted != labels
    positives, negatives = values[labels], values[~labels]
    return {
        "threshold": threshold,
        "point_errors": int(point_errors),
        "case_errors": int(len(np.unique(source["case_indices"][bad]))),
        "cases": int(source["case_count"]),
        "points": int(len(labels)),
        "positive_min": float(positives.min()),
        "negative_max": float(negatives.max()),
        "gap": float(positives.min() - negatives.max()),
        "invalid_base": int(np.count_nonzero(base.numpy() <= 1e-8)),
        "horizontal_min": float(horizontal.numpy().min()),
        "vertical_min": float(vertical.numpy().min()),
    }


def initial_pair(rank: int) -> tuple[np.ndarray, np.ndarray]:
    factor = common.authority_factor().astype(np.float64)
    kernel = factor @ factor.T
    u, values, vt = np.linalg.svd(kernel, full_matrices=False)
    root = np.sqrt(values[:rank])
    left = (u[:, :rank] * root).astype(np.float32)
    right = (vt[:rank].T * root).astype(np.float32)
    left[20:] = 0
    right[20:] = 0
    return left, right


def train_one(
    initial_left: np.ndarray,
    initial_right: np.ndarray,
    source: dict[str, np.ndarray],
    steps: int,
    learning_rate: float,
    seed: int,
    axis_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    torch.manual_seed(seed)
    left = torch.nn.Parameter(torch.from_numpy(initial_left.copy()))
    right = torch.nn.Parameter(torch.from_numpy(initial_right.copy()))
    threshold = torch.nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    axis_threshold = torch.nn.Parameter(torch.tensor(0.14, dtype=torch.float32))
    optimizer = torch.optim.Adam([left, right, threshold, axis_threshold], lr=learning_rate)
    data = common.torch_data(source)
    labels = torch.from_numpy(source["labels"].astype(np.float32))
    h_labels = torch.from_numpy(source["horizontal_labels"].astype(np.float32))
    v_labels = torch.from_numpy(source["vertical_labels"].astype(np.float32))
    left_norm, right_norm = float(np.linalg.norm(initial_left)), float(np.linalg.norm(initial_right))
    best_left, best_right = initial_left.copy(), initial_right.copy()
    best = evaluate(best_left, best_right, source)
    best["step"] = 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        ratio, horizontal, vertical, base = ratios(left, right, data)
        signs = labels * 2 - 1
        losses = nnf.softplus(-80 * signs * (ratio - threshold)) / 80
        loss = losses.mean() + 0.08 * torch.logsumexp(40 * losses, dim=0) / 40
        if axis_weight:
            h_signs, v_signs = h_labels * 2 - 1, v_labels * 2 - 1
            axis_losses = torch.cat((
                nnf.softplus(-40 * h_signs * (horizontal - axis_threshold)) / 40,
                nnf.softplus(-40 * v_signs * (vertical - axis_threshold)) / 40,
            ))
            loss = loss + axis_weight * (
                axis_losses.mean() + 0.03 * torch.logsumexp(30 * axis_losses, dim=0) / 30
            )
        loss = loss + 0.02 * nnf.softplus(1e-3 - base).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([left, right, threshold, axis_threshold], 10)
        optimizer.step()
        with torch.no_grad():
            left[20:].zero_()
            right[20:].zero_()
            left.mul_(left_norm / left.norm().clamp_min(1e-8))
            right.mul_(right_norm / right.norm().clamp_min(1e-8))
            threshold.clamp_(-1, 2)
            axis_threshold.clamp_(-1, 2)
        if step % 50 == 0 or step == steps:
            candidate_left = left.detach().numpy().copy()
            candidate_right = right.detach().numpy().copy()
            report = evaluate(candidate_left, candidate_right, source)
            report["step"] = step
            if (report["point_errors"], -report["gap"]) < (best["point_errors"], -best["gap"]):
                best_left, best_right, best = candidate_left, candidate_right, report
            if report["point_errors"] == 0 and report["gap"] > 1e-4:
                break
    return best_left, best_right, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=0.006)
    parser.add_argument("--axis-weight", type=float, default=0.2)
    args = parser.parse_args()
    with np.load(HERE / "rank7_noise_train.npz") as loaded:
        train = {name: loaded[name] for name in loaded.files}
    with np.load(HERE / "rank7_noise_validation.npz") as loaded:
        validation = {name: loaded[name] for name in loaded.files}
    base_left, base_right = initial_pair(args.rank)
    rng = np.random.default_rng(192_430_001)
    rows = []
    winner = None
    for restart in range(args.restarts):
        scale = 0.0 if restart == 0 else 0.03 * restart
        initial_left = base_left + rng.normal(0, scale, base_left.shape).astype(np.float32)
        initial_right = base_right + rng.normal(0, scale, base_right.shape).astype(np.float32)
        initial_left[20:] = 0
        initial_right[20:] = 0
        left, right, train_report = train_one(
            initial_left, initial_right, train, args.steps, args.learning_rate,
            192_430_010 + restart, args.axis_weight,
        )
        validation_report = evaluate(left, right, validation)
        row = {"restart": restart, "train": train_report, "validation": validation_report}
        rows.append(row)
        print(json.dumps(row), flush=True)
        key = (validation_report["point_errors"], train_report["point_errors"], -validation_report["gap"])
        if winner is None or key < winner[0]:
            winner = (key, left, right, train_report, validation_report)
            np.savez(HERE / f"general_rank{args.rank}_best.npz", left=left, right=right)
        (HERE / f"general_rank{args.rank}_training.json").write_text(json.dumps({
            "args": vars(args), "runs": rows,
            "best": {"train": winner[3], "validation": winner[4]},
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
