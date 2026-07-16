#!/usr/bin/env python3
"""Search lower-rank task270 renderers on the complete semantic domain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def domain() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    names = ("O", "B", "C", "P")
    codes = {
        "O": (0.0, 0.0),
        "B": (1.0, 1.0),
        "C": (-0.25, 0.5),
        "P": (-1.0, 0.75),
    }
    states = [("O/O", codes["O"])]
    states.extend(
        (f"{left}/{right}", (codes[left][0], codes[right][1]))
        for left in names[1:]
        for right in names[1:]
        if (left, right) != ("C", "C")
    )
    code_tensor = torch.tensor([value for _, value in states], dtype=torch.float64)
    targets = []
    valid = []
    for row_index, (row_name, _) in enumerate(states):
        row_states = row_name.split("/")
        for col_index, (col_name, _) in enumerate(states):
            col_states = col_name.split("/")
            objects = []
            if "O" not in row_states + col_states:
                for flower, (center, petal) in enumerate(((2, 3), (1, 7))):
                    pair = (row_states[flower], col_states[flower])
                    if pair == ("C", "C"):
                        objects.append(center)
                    elif pair in {("C", "P"), ("P", "C")}:
                        objects.append(petal)
            if len(objects) > 1:
                continue
            target = -torch.ones(10, dtype=torch.float64)
            if objects:
                target[objects[0]] = 1.0
            elif "O" not in row_states + col_states:
                target[0] = 1.0
            valid.append((row_index, col_index))
            targets.append(target)
    assert len(valid) == 79
    return code_tensor, torch.tensor(valid, dtype=torch.long), torch.stack(targets)


def scores(theta: torch.Tensor, kernel: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    # Normalizing A columns removes an otherwise severe sixth-power scale
    # ambiguity.  Kernel magnitude retains arbitrary score scale.
    basis = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
    # Each of the six repeated renderer indices links R, C, and A together:
    # z_q(i,j) = sum_d row_i[d] * col_j[d] * A[d,q].
    joint = torch.einsum("id,jd,bdq->bijq", codes, codes, basis).pow(6)
    return torch.einsum("bijq,bqk->bijk", joint, kernel)


def fp16_check(theta: np.ndarray, kernel: np.ndarray, codes: np.ndarray,
               valid: np.ndarray, target: np.ndarray) -> dict[str, float | int | bool]:
    basis = np.stack((np.cos(theta), np.sin(theta)), axis=0).astype(np.float16)
    kernel16 = kernel.astype(np.float16)
    # Mirror the renderer's fp16 multiply/add domain conservatively.  NumPy
    # evaluates the contractions in fp16 when both inputs are fp16.
    joint = np.einsum("id,jd,dq->ijq", codes.astype(np.float16),
                      codes.astype(np.float16), basis,
                      dtype=np.float16, optimize=False)
    joint = (joint ** np.int16(6)).astype(np.float16)
    raw = np.einsum("ijq,qk->ijk", joint, kernel16,
                    dtype=np.float16, optimize=False)
    got = np.stack([raw[i, j] for i, j in valid])
    positive = target > 0
    wrong = (positive & (got <= 0)) | (~positive & (got > 0))
    intended = got[positive]
    unexpected = got[~positive]
    return {
        "correct": bool(not np.any(wrong)),
        "wrong_signs": int(np.count_nonzero(wrong)),
        "min_signed": float(np.min(intended)),
        "max_unexpected": float(np.max(unexpected)),
        "max_abs": float(np.max(np.abs(got))),
        "finite": bool(np.isfinite(got).all()),
        "A": basis.tolist(),
        "K": kernel16.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=270195)
    parser.add_argument("--pos-weight", type=float, default=12.0)
    parser.add_argument("--lr", type=float, default=0.025)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    codes, valid, target = domain()
    theta = torch.nn.Parameter(torch.rand(args.batch, args.rank, dtype=torch.float64) * math.pi)
    kernel = torch.nn.Parameter(torch.randn(args.batch, args.rank, 10, dtype=torch.float64) * 0.1)
    optimizer = torch.optim.Adam([theta, kernel], lr=args.lr)
    best = None
    rows = valid[:, 0]
    cols = valid[:, 1]
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        raw = scores(theta, kernel, codes)[:, rows, cols, :]
        positive = target.unsqueeze(0) > 0
        # Smooth hinge plus a mild kernel penalty.  A high target margin makes
        # the subsequent fp16 quantization robust.
        violations = torch.where(positive, torch.relu(1.0 - raw), torch.relu(raw))
        weights = torch.where(positive, args.pos_weight, 1.0)
        loss_each = (weights * violations.square()).mean(dim=(1, 2))
        loss = loss_each.mean() + 1e-8 * kernel.square().mean()
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == args.steps:
            with torch.no_grad():
                intended = torch.where(positive, raw, torch.inf)
                mins = intended.amin(dim=(1, 2))
                wrong = ((positive & (raw <= 0)) | (~positive & (raw > 0))).sum(dim=(1, 2))
                metric = -wrong.to(torch.float64) * 1e9 + mins
                idx = int(torch.argmax(metric))
                candidate = fp16_check(
                    theta[idx].cpu().numpy(),
                    kernel[idx].cpu().numpy(),
                    codes.cpu().numpy(),
                    valid.cpu().numpy(),
                    target.cpu().numpy(),
                )
                summary = {
                    "step": step,
                    "best_float_min_signed": float(mins[idx]),
                    "best_float_loss": float(loss_each[idx]),
                    "best_float_wrong": int(wrong[idx]),
                    "fp16_wrong": candidate["wrong_signs"],
                    "fp16_min_signed": candidate["min_signed"],
                }
                print(json.dumps(summary), flush=True)
                if best is None or (candidate["wrong_signs"], -candidate["min_signed"]) < (
                    best["wrong_signs"], -best["min_signed"]
                ):
                    best = {**candidate, **summary, "rank": args.rank, "seed": args.seed}
                    (HERE / "search").mkdir(parents=True, exist_ok=True)
                    (HERE / "search" / f"rank{args.rank}_best.json").write_text(
                        json.dumps(best, indent=2) + "\n"
                    )
                if candidate["correct"] and candidate["min_signed"] >= 0.25:
                    break
    print(json.dumps(best, indent=2))
    return 0 if best and best["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
