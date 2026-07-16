#!/usr/bin/env python3
"""Search lower CP rank for task229's output-only symmetric Einsum classifier."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import zipfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
AUTHORITY = ROOT / "submission_base_8011.05.zip"
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for value in range(total + 1):
        for tail in compositions(total - value, parts - 1):
            yield (value, *tail)


def dataset() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    powers = []
    cell_colors = []
    desired = []
    for count_colors in (2, 3, 4):
        for chosen in itertools.combinations(COLORS, count_colors):
            for counts in compositions(9, count_colors):
                maximum = max(counts)
                if counts.count(maximum) != 1:
                    continue
                mode = chosen[counts.index(maximum)]
                p = np.zeros(10, dtype=np.float64)
                for color, count in zip(chosen, counts):
                    p[color] = (count / 9.0) ** 17
                for color, count in zip(chosen, counts):
                    if count == 0:
                        continue
                    powers.append(p)
                    cell_colors.append(color)
                    desired.append(color if color == mode else 5)
    return (
        torch.tensor(np.asarray(powers), dtype=torch.float64),
        torch.tensor(cell_colors, dtype=torch.long),
        torch.tensor(desired, dtype=torch.long),
    )


def logits(a: torch.Tensor, p: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    # raw[e,o] = sum_r A[r,o] * A[r,cell[e]] * sum_d p[e,d]*A[r,d]
    q = p @ a.T
    ac = a[:, cell].T
    return (q * ac) @ a


def accuracy(raw: torch.Tensor, target: torch.Tensor) -> tuple[float, float, int]:
    positive = raw > 0
    wanted = torch.zeros_like(positive)
    wanted[torch.arange(len(target)), target] = True
    exact = (positive == wanted).all(dim=1)
    signed = torch.where(wanted, raw, -raw)
    return float(exact.double().mean()), float(signed.min()), int((~exact).sum())


def train(rank: int, steps: int, restarts: int, seed: int) -> dict:
    p, cell, target = dataset()
    with zipfile.ZipFile(AUTHORITY) as archive:
        model = onnx.load_from_string(archive.read("task229.onnx"))
    original = numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    y = torch.full((len(target), 10), -1.0, dtype=torch.float64)
    y[torch.arange(len(target)), target] = 1.0
    best = None
    for restart in range(restarts):
        generator = torch.Generator().manual_seed(seed + restart)
        if rank < 4 and restart < 4:
            start = np.delete(original, restart, axis=0)[:rank]
            start = torch.tensor(start, dtype=torch.float64)
        else:
            start = torch.randn((rank, 10), generator=generator, dtype=torch.float64)
        a = torch.nn.Parameter(start)
        optimizer = torch.optim.Adam([a], lr=0.025)
        local_best = None
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            raw = logits(a, p, cell)
            scale = 2.0 / (raw.detach().abs().median() + 1e-6)
            loss = torch.nn.functional.softplus(-y * raw * scale).mean()
            loss = loss + 1e-7 * (a * a).mean()
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step + 1 == steps:
                with torch.no_grad():
                    raw = logits(a, p, cell)
                    acc, margin, wrong = accuracy(raw, target)
                    row = {
                        "restart": restart, "step": step, "accuracy": acc,
                        "margin": margin, "wrong": wrong, "loss": float(loss),
                        "A": a.detach().cpu().numpy().tolist(),
                    }
                    if local_best is None or (acc, margin) > (
                        local_best["accuracy"], local_best["margin"]
                    ):
                        local_best = row
                    if acc == 1.0 and margin > 1e-7:
                        break
        if best is None or (local_best["accuracy"], local_best["margin"]) > (
            best["accuracy"], best["margin"]
        ):
            best = local_best
        print(json.dumps({k: local_best[k] for k in local_best if k != "A"}), flush=True)
    assert best is not None
    payload = {
        "rank": rank, "dataset_examples": len(target), "steps": steps,
        "restarts": restarts, "seed": seed, "best": best,
    }
    (HERE / f"task229_rank{rank}_train.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps({k: best[k] for k in best if k != "A"}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=304229)
    args = parser.parse_args()
    train(args.rank, args.steps, args.restarts, args.seed)


if __name__ == "__main__":
    main()
