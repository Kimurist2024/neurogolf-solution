#!/usr/bin/env python3
"""Hard-negative gap refinement for the task192 rank-8 PSD kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nnf

import train_rank7_noise as common


HERE = Path(__file__).resolve().parent


def combine(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    point_names = (
        "rows", "cols", "inside_rows", "inside_cols", "row_indices", "col_indices",
        "labels", "horizontal_labels", "vertical_labels", "case_indices",
    )
    result = {}
    offset = int(first["case_count"])
    for name in point_names:
        right = second[name] + offset if name == "case_indices" else second[name]
        result[name] = np.concatenate((first[name], right), axis=0)
    result["case_count"] = np.asarray(offset + int(second["case_count"]), dtype=np.int64)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--hard", type=int, default=128)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--axis-floor", type=float, default=0.16)
    parser.add_argument("--axis-weight", type=float, default=0.15)
    args = parser.parse_args()
    with np.load(HERE / "rank7_noise_train.npz") as loaded:
        first = {name: loaded[name] for name in loaded.files}
    with np.load(HERE / "rank7_noise_validation.npz") as loaded:
        second = {name: loaded[name] for name in loaded.files}
    source = combine(first, second)
    strict_path = HERE / "rank8_noise_strict_seed777192.npz"
    if strict_path.exists():
        with np.load(strict_path) as loaded:
            strict = {name: loaded[name] for name in loaded.files}
        source = combine(source, strict)
    for cegis_path in sorted(HERE.glob("rank8_noise_cegis_*.npz")):
        with np.load(cegis_path) as loaded:
            cegis = {name: loaded[name] for name in loaded.files}
        source = combine(source, cegis)
    data = common.torch_data(source)
    labels = torch.from_numpy(source["labels"])
    horizontal_labels = torch.from_numpy(source["horizontal_labels"])
    vertical_labels = torch.from_numpy(source["vertical_labels"])
    gap_path = HERE / "rank8_noise_gap_factor.npy"
    initial = np.load(gap_path if gap_path.exists() else HERE / "rank8_noise_best_factor.npy").astype(np.float32)
    initial_norm = float(np.linalg.norm(initial))
    rng = np.random.default_rng(192_432_001)
    rows = []
    winner = None
    for restart in range(args.restarts):
        start = initial.copy()
        if restart:
            start += rng.normal(0, 0.005 * restart, start.shape).astype(np.float32)
            start[20:] = 0
            start *= initial_norm / np.linalg.norm(start)
        factor = torch.nn.Parameter(torch.from_numpy(start))
        optimizer = torch.optim.Adam([factor], lr=args.learning_rate)
        best_factor = start.copy()
        best = common.evaluate(best_factor, source)
        best["step"] = 0
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            ratio, horizontal, vertical = common.ratios(factor, data)
            positives = ratio[labels]
            negatives = ratio[~labels]
            hard = min(args.hard, len(positives), len(negatives))
            low_positive = torch.topk(positives, hard, largest=False).values
            high_negative = torch.topk(negatives, hard, largest=True).values
            # Pair all hard extremes.  The smooth barrier remains active after
            # the observed gap becomes positive and therefore grows margin.
            pairwise = high_negative[:, None] - low_positive[None, :] + 0.002
            loss = nnf.softplus(pairwise * 80).mean() / 80
            # Maintain positive, well-conditioned inside normalizers.
            local_axis_values = torch.cat((
                horizontal[horizontal_labels], vertical[vertical_labels],
            ))
            loss = loss + args.axis_weight * (
                nnf.softplus((args.axis_floor - local_axis_values) * 40).mean() / 40
            )
            loss = loss + 1e-4 * (factor - torch.from_numpy(initial)).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([factor], 5)
            optimizer.step()
            with torch.no_grad():
                factor[20:].zero_()
                factor.mul_(initial_norm / factor.norm().clamp_min(1e-8))
            if step % 50 == 0 or step == args.steps:
                candidate = factor.detach().numpy().copy()
                report = common.evaluate(candidate, source)
                report["step"] = step
                if (report["point_errors"], -report["gap"]) < (best["point_errors"], -best["gap"]):
                    best_factor, best = candidate, report
                if (
                    step >= 1_000
                    and report["point_errors"] == 0
                    and report["gap"] > 0.001
                    and report["base_axis_horizontal_positive_min"] >= args.axis_floor * 0.8
                    and report["base_axis_vertical_positive_min"] >= args.axis_floor * 0.8
                ):
                    break
        row = {"restart": restart, "combined": best}
        rows.append(row)
        print(json.dumps(row), flush=True)
        key = (best["point_errors"], -best["gap"])
        if winner is None or key < winner[0]:
            winner = (key, best_factor, best)
            np.save(HERE / "rank8_noise_gap_factor.npy", best_factor)
        (HERE / "rank8_noise_gap_training.json").write_text(json.dumps({
            "args": vars(args), "runs": rows, "best": winner[2],
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
