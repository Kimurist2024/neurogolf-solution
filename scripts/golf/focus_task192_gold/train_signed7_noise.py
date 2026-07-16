#!/usr/bin/env python3
"""Train a learned signed symmetric rank-7 kernel for task192 noise cells."""

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
    factor: torch.Tensor, signs: torch.Tensor, data: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row_projection = data["rows"] @ factor
    col_projection = data["cols"] @ factor
    inside_row_projection = data["inside_rows"] @ factor
    inside_col_projection = data["inside_cols"] @ factor
    at_col = factor[data["col_indices"]] * signs
    at_row = factor[data["row_indices"]] * signs
    horizontal = (row_projection * at_col).sum(dim=1)
    vertical = (col_projection * at_row).sum(dim=1)
    horizontal_base = (inside_row_projection * at_col).sum(dim=1)
    vertical_base = (inside_col_projection * at_row).sum(dim=1)
    base = horizontal_base * vertical_base
    ratio = horizontal * vertical / base.clamp_min(1e-8)
    return ratio, horizontal / horizontal_base.clamp_min(1e-8), vertical / vertical_base.clamp_min(1e-8), base


def evaluate(factor: np.ndarray, signs: np.ndarray, source: dict[str, np.ndarray]) -> dict[str, object]:
    with torch.no_grad():
        ratio, horizontal, vertical, base = ratios(
            torch.from_numpy(factor), torch.from_numpy(signs), common.torch_data(source)
        )
    values = ratio.numpy().astype(np.float64)
    labels = source["labels"]
    threshold, point_errors = common.best_threshold(values, labels)
    bad = (values > threshold) != labels
    return {
        "threshold": threshold,
        "point_errors": int(point_errors),
        "case_errors": int(len(np.unique(source["case_indices"][bad]))),
        "cases": int(source["case_count"]),
        "points": int(len(labels)),
        "positive_min": float(values[labels].min()),
        "negative_max": float(values[~labels].max()),
        "gap": float(values[labels].min() - values[~labels].max()),
        "invalid_base": int(np.count_nonzero(base.numpy() <= 1e-8)),
        "horizontal_min": float(horizontal.numpy().min()),
        "vertical_min": float(vertical.numpy().min()),
    }


def initial_factor(negative: int, variant: int) -> tuple[np.ndarray, np.ndarray]:
    rank = 7
    positive = rank - negative
    current = common.authority_factor().astype(np.float64)
    current_kernel = current @ current.T
    values, vectors = np.linalg.eigh(current_kernel)
    pos_order = np.argsort(values)[::-1][:positive]
    pos_factor = vectors[:, pos_order] * np.sqrt(np.maximum(values[pos_order], 0))

    target = np.eye(30) + np.eye(30, k=1) + np.eye(30, k=-1)
    target_values, target_vectors = np.linalg.eigh(target)
    neg_order = np.argsort(target_values)[:negative]
    # Start correction modes small enough to preserve the good PSD authority,
    # but nonzero so gradient descent can grow them.
    scale = (0.03, 0.08, 0.16, 0.3)[variant % 4]
    neg_factor = target_vectors[:, neg_order] * np.sqrt(np.abs(target_values[neg_order])) * scale
    factor = np.concatenate((pos_factor, neg_factor), axis=1).astype(np.float32)
    factor[20:] = 0
    signs = np.asarray([1.0] * positive + [-1.0] * negative, dtype=np.float32)
    return factor, signs


def train_one(
    initial: np.ndarray,
    signs_np: np.ndarray,
    source: dict[str, np.ndarray],
    steps: int,
    learning_rate: float,
    axis_weight: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    torch.manual_seed(seed)
    factor = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    signs = torch.from_numpy(signs_np)
    threshold = torch.nn.Parameter(torch.tensor(0.055, dtype=torch.float32))
    axis_threshold = torch.nn.Parameter(torch.tensor(0.14, dtype=torch.float32))
    optimizer = torch.optim.Adam([factor, threshold, axis_threshold], lr=learning_rate)
    data = common.torch_data(source)
    labels = torch.from_numpy(source["labels"].astype(np.float32))
    h_labels = torch.from_numpy(source["horizontal_labels"].astype(np.float32))
    v_labels = torch.from_numpy(source["vertical_labels"].astype(np.float32))
    target_norm = float(np.linalg.norm(initial))
    best_factor = initial.copy()
    best = evaluate(best_factor, signs_np, source)
    best["step"] = 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        ratio, horizontal, vertical, base = ratios(factor, signs, data)
        y = labels * 2 - 1
        losses = nnf.softplus(-80 * y * (ratio - threshold)) / 80
        loss = losses.mean() + 0.1 * torch.logsumexp(40 * losses, dim=0) / 40
        if axis_weight:
            hy, vy = h_labels * 2 - 1, v_labels * 2 - 1
            axis_losses = torch.cat((
                nnf.softplus(-40 * hy * (horizontal - axis_threshold)) / 40,
                nnf.softplus(-40 * vy * (vertical - axis_threshold)) / 40,
            ))
            loss = loss + axis_weight * (
                axis_losses.mean() + 0.03 * torch.logsumexp(30 * axis_losses, dim=0) / 30
            )
        loss = loss + 0.02 * nnf.softplus(1e-3 - base).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([factor, threshold, axis_threshold], 10)
        optimizer.step()
        with torch.no_grad():
            factor[20:].zero_()
            factor.mul_(target_norm / factor.norm().clamp_min(1e-8))
            threshold.clamp_(-1, 2)
            axis_threshold.clamp_(-1, 2)
        if step % 50 == 0 or step == steps:
            candidate = factor.detach().numpy().copy()
            report = evaluate(candidate, signs_np, source)
            report["step"] = step
            if (report["point_errors"], -report["gap"]) < (best["point_errors"], -best["gap"]):
                best_factor, best = candidate, report
            if report["point_errors"] == 0 and report["gap"] > 1e-4:
                break
    return best_factor, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative", type=int, default=1)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=0.006)
    parser.add_argument("--axis-weight", type=float, default=0.2)
    args = parser.parse_args()
    with np.load(HERE / "rank7_noise_train.npz") as loaded:
        train = {name: loaded[name] for name in loaded.files}
    with np.load(HERE / "rank7_noise_validation.npz") as loaded:
        validation = {name: loaded[name] for name in loaded.files}
    rows = []
    winner = None
    for variant in range(args.variants):
        initial, signs = initial_factor(args.negative, variant)
        factor, train_report = train_one(
            initial, signs, train, args.steps, args.learning_rate,
            args.axis_weight, 192_431_000 + args.negative * 10 + variant,
        )
        validation_report = evaluate(factor, signs, validation)
        row = {"variant": variant, "signs": signs.tolist(), "train": train_report, "validation": validation_report}
        rows.append(row)
        print(json.dumps(row), flush=True)
        key = (validation_report["point_errors"], train_report["point_errors"], -validation_report["gap"])
        if winner is None or key < winner[0]:
            winner = (key, factor, signs, train_report, validation_report)
            np.savez(HERE / f"signed_rank7_n{args.negative}_best.npz", factor=factor, signs=signs)
        (HERE / f"signed_rank7_n{args.negative}_training.json").write_text(json.dumps({
            "args": vars(args), "runs": rows,
            "best": {"train": winner[3], "validation": winner[4]},
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
