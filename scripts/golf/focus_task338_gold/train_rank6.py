#!/usr/bin/env python3
"""Train a truthful rank-6 repair of the archived task338 rank-5 Einsum.

The graph formula is unchanged.  Only its shared 30x30 relation matrix is
upgraded from rank 5 to rank 6 and trained against generator-produced boxes.
Rank 6 has cost 394 (34 fixed params + 30*6 + 6*30), below authority 403.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = (
    ROOT
    / "scripts/golf/loop_7999_13/lane_archive_loose_sweep/"
    / "task338_r01_static334.onnx"
)


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUPPORT = import_path(
    "task338_rank6_support",
    ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py",
)
SUPPORT.THRESHOLD = 1.0
SUPPORT.FRESH_PER_SEED = 2_000
SUPPORT.SUPPORT.POLICY_THRESHOLD = 1.0
SUPPORT.SUPPORT.FRESH_PER_SEED = 2_000


def arrays(examples: list[dict[str, Any]]) -> tuple[torch.Tensor, ...]:
    converted = [SUPPORT.SUPPORT.scoring.convert_to_numpy(row) for row in examples]
    converted = [row for row in converted if row is not None]
    a0 = np.stack(
        [row["input"][0, 0] + row["input"][0, 2] for row in converted]
    ).astype(np.float32)
    a1 = np.stack([row["input"][0, 2] for row in converted]).astype(np.float32)
    target = np.stack([row["output"][0, 3] > 0 for row in converted])
    return (
        torch.from_numpy(a0),
        torch.from_numpy(a1),
        torch.from_numpy(target),
    )


def h_transform(mask: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    triangle = torch.einsum(
        "jw,wk,jk->wjk", matrix, matrix, matrix
    )
    return torch.einsum("bhj,wjk,bhk->bhw", mask, triangle, mask)


def raw_score(
    a0: torch.Tensor,
    a1: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    log_epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = u @ v
    h0 = h_transform(a0, matrix)
    h1 = h_transform(a1, matrix)
    v0 = h_transform(a0.transpose(1, 2), matrix).transpose(1, 2)
    v1 = h_transform(a1.transpose(1, 2), matrix).transpose(1, 2)
    positive_term = (a0 - a1) * h1 * v1
    bias_term = torch.exp(log_epsilon) * a0 * h0 * v0
    score = positive_term - bias_term
    normalized = score / (positive_term.abs() + bias_term.abs() + 1e-8)
    return score, normalized


def balanced_loss(
    normalized: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    pos = normalized[target & valid]
    neg = normalized[(~target) & valid]
    pos_loss = F.softplus(-12.0 * pos).mean() if pos.numel() else normalized.sum() * 0
    neg_loss = F.softplus(12.0 * neg).mean() if neg.numel() else normalized.sum() * 0
    return pos_loss + neg_loss


@torch.no_grad()
def evaluate(
    dataset: tuple[torch.Tensor, ...],
    u: torch.Tensor,
    v: torch.Tensor,
    log_epsilon: torch.Tensor,
    batch_size: int = 128,
) -> dict[str, Any]:
    a0, a1, target = dataset
    wrong_cells = 0
    wrong_cases = 0
    minimum_positive = float("inf")
    maximum_nonpositive = -float("inf")
    for start in range(0, len(a0), batch_size):
        aa0 = a0[start : start + batch_size]
        aa1 = a1[start : start + batch_size]
        yy = target[start : start + batch_size]
        score, _ = raw_score(aa0, aa1, u, v, log_epsilon)
        valid = aa0 > 0
        pred = score > 0
        wrong = (pred != yy) & valid
        wrong_cells += int(wrong.sum())
        wrong_cases += int(wrong.flatten(1).any(1).sum())
        positive = score[pred & valid]
        nonpositive = score[(~pred) & valid]
        if positive.numel():
            minimum_positive = min(minimum_positive, float(positive.min()))
        if nonpositive.numel():
            maximum_nonpositive = max(maximum_nonpositive, float(nonpositive.max()))
    return {
        "cases": len(a0),
        "wrong_cells": wrong_cells,
        "wrong_cases": wrong_cases,
        "case_accuracy": 1.0 - wrong_cases / max(1, len(a0)),
        "minimum_positive": None if minimum_positive == float("inf") else minimum_positive,
        "maximum_nonpositive": None
        if maximum_nonpositive == -float("inf")
        else maximum_nonpositive,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=338006)
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    args = parser.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    known_examples, known_counts = SUPPORT.SUPPORT.known_cases(338)
    train_examples, train_generation = SUPPORT.SUPPORT.fresh_cases(
        338, args.seed * 10 + 1, task_map
    )
    validation_examples, validation_generation = SUPPORT.SUPPORT.fresh_cases(
        338, args.seed * 10 + 2, task_map
    )
    known = arrays(known_examples)
    training = arrays(train_examples)
    validation = arrays(validation_examples)

    source = onnx.load(str(SOURCE))
    init = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in source.graph.initializer
    }
    u0 = np.zeros((30, 6), dtype=np.float32)
    v0 = np.zeros((6, 30), dtype=np.float32)
    u0[:, :5] = init["U"]
    v0[:5, :] = init["V"]
    u0[:, 5] = np.random.normal(0, 0.01, 30)
    v0[5, :] = np.random.normal(0, 0.01, 30)
    u = torch.nn.Parameter(torch.from_numpy(u0))
    v = torch.nn.Parameter(torch.from_numpy(v0))
    log_epsilon = torch.nn.Parameter(
        torch.tensor(float(np.log(-init["K"][0, 0])), dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([u, v, log_epsilon], lr=args.lr)

    # Mix the immutable local gold set into training so a fresh improvement can
    # never trade away known exactness.
    train_a0 = torch.cat([training[0], known[0].repeat(2, 1, 1)])
    train_a1 = torch.cat([training[1], known[1].repeat(2, 1, 1)])
    train_y = torch.cat([training[2], known[2].repeat(2, 1, 1)])
    count = len(train_a0)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for step in range(1, args.steps + 1):
        index = torch.randint(0, count, (args.batch_size,))
        a0 = train_a0[index]
        a1 = train_a1[index]
        target = train_y[index]
        _, normalized = raw_score(a0, a1, u, v, log_epsilon)
        loss = balanced_loss(normalized, target, a0 > 0)
        # Keep the low-rank factorization numerically bounded without changing
        # its sign capacity.
        loss = loss + 1e-6 * ((u @ v).square().mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([u, v, log_epsilon], 10.0)
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            known_eval = evaluate(known, u, v, log_epsilon)
            validation_eval = evaluate(validation, u, v, log_epsilon)
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "epsilon": float(torch.exp(log_epsilon).detach()),
                "known": known_eval,
                "validation": validation_eval,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            objective = (
                known_eval["wrong_cells"] + validation_eval["wrong_cells"],
                known_eval["wrong_cases"] + validation_eval["wrong_cases"],
            )
            if best is None or objective < tuple(best["objective"]):
                best = {
                    "objective": list(objective),
                    "step": step,
                    "u": u.detach().cpu().numpy().copy(),
                    "v": v.detach().cpu().numpy().copy(),
                    "epsilon": float(torch.exp(log_epsilon).detach()),
                    "known": known_eval,
                    "validation": validation_eval,
                }
                np.savez(
                    HERE / f"rank6_best_seed{args.seed}.npz",
                    u=best["u"],
                    v=best["v"],
                    epsilon=np.asarray(best["epsilon"], dtype=np.float32),
                )
            if objective == (0, 0):
                break

    assert best is not None
    payload = {
        "task": 338,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_cost": 334,
        "authority_cost": 403,
        "candidate_rank": 6,
        "candidate_cost": 394,
        "seed": args.seed,
        "steps_requested": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "known_counts": known_counts,
        "train_generation": train_generation,
        "validation_generation": validation_generation,
        "best": {key: value for key, value in best.items() if key not in {"u", "v"}},
        "history": history,
    }
    (HERE / f"training_seed{args.seed}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
