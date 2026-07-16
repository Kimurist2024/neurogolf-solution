#!/usr/bin/env python3
"""Retrain task023's clean 6x6 QLinearConv ranker without changing its cost.

The task generator is non-injective, so this lane is explicitly empirical.  It
keeps the existing, structurally clean graph and changes only ``score_W_q``.
Known fixtures are included in every training epoch and remain a hard
selection gate when candidates are written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as functional
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
KNOWN_PATH = ROOT / "inputs" / "neurogolf-2026" / "task023.json"
SOURCE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_clean95_all/candidates"
    / "task023_9a2b78138891_cost1541.onnx"
)
sys.path.insert(0, str(TASK_DIR))


def onehot(grid: list[list[int]]) -> np.ndarray:
    values = np.asarray(grid, dtype=np.int64)
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    rows, cols = np.indices(values.shape)
    result[0, values, rows, cols] = 1.0
    return result


def case_features(example: dict[str, list[list[int]]]) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the 36 convolution patches, target mask, and active TopK count."""
    grid = np.asarray(example["input"], dtype=np.int8)
    output = np.asarray(example["output"], dtype=np.int8)
    gray = (grid[:8, 1:9] == 5).astype(np.float32)
    padded = np.pad(gray, ((2, 1), (2, 1)))
    patches = np.stack(
        [padded[row : row + 6, col : col + 6].reshape(-1) for row in range(6) for col in range(6)]
    )
    cyan = output == 8
    target = np.zeros(36, dtype=bool)
    for row in range(output.shape[0] - 1):
        for col in range(output.shape[1] - 1):
            if bool(cyan[row : row + 2, col : col + 2].all()):
                score_col = col - 1
                if 0 <= row < 6 and 0 <= score_col < 6:
                    target[row * 6 + score_col] = True
    active = 2 if grid.shape[1] == 9 else 3
    if int(target.sum()) != active:
        raise ValueError(
            f"unexpected target count {target.sum()} for input shape {grid.shape}"
        )
    return patches, target, active


def generated_cases(count: int, seed: int) -> list[dict[str, list[list[int]]]]:
    common = importlib.import_module("common")
    generator = importlib.import_module("task_150deff5")
    result = []
    for index in range(count):
        current = seed + index
        random.seed(current)
        common.random.seed(current)
        result.append(generator.generate())
    return result


def known_cases() -> list[dict[str, list[list[int]]]]:
    data = json.loads(KNOWN_PATH.read_text())
    return [item for split in ("train", "test", "arc-gen") for item in data.get(split, [])]


def make_dataset(cases: list[dict[str, list[list[int]]]]) -> tuple[np.ndarray, np.ndarray]:
    rows = [case_features(case) for case in cases]
    return np.stack([row[0] for row in rows]), np.stack([row[1] for row in rows])


def exact_rate(features: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> tuple[int, int]:
    raw = np.einsum("nif,f->ni", features.astype(np.int16), weights.astype(np.int16))
    scores = np.clip(raw, 0, 255)
    right = 0
    for score, target in zip(scores, targets):
        count = int(target.sum())
        selected = np.argsort(-score, kind="stable")[:count]
        right += bool(target[selected].all())
    return right, len(features)


def replace_weights(model: onnx.ModelProto, weights: np.ndarray) -> None:
    for initializer in model.graph.initializer:
        if initializer.name == "score_W_q":
            value = weights.astype(np.int8).reshape(1, 1, 6, 6)
            initializer.CopyFrom(numpy_helper.from_array(value, initializer.name))
            return
    raise KeyError("score_W_q")


def ort_rate(model: onnx.ModelProto, cases: list[dict[str, list[list[int]]]], mode: str) -> tuple[int, int]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    right = 0
    for example in cases:
        raw = session.run(["output"], {"input": onehot(example["input"])})[0]
        right += bool(np.array_equal(raw > 0, onehot(example["output"]).astype(bool)))
    return right, len(cases)


def quantizations(weights: np.ndarray) -> list[np.ndarray]:
    max_abs = max(float(np.abs(weights).max()), 1.0e-6)
    scales = np.geomspace(8.0 / max_abs, 127.0 / max_abs, 40)
    result: list[np.ndarray] = []
    seen: set[bytes] = set()
    for scale in scales:
        value = np.clip(np.rint(weights * scale), -127, 127).astype(np.int8)
        key = value.tobytes()
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=24000)
    parser.add_argument("--valid", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=150_023_800)
    args = parser.parse_args()

    torch.set_num_threads(4)
    known = known_cases()
    fresh_train = generated_cases(args.train, args.seed)
    fresh_valid = generated_cases(args.valid, args.seed + 10_000_000)
    known_x, known_y = make_dataset(known)
    fresh_x, fresh_y = make_dataset(fresh_train)
    valid_x, valid_y = make_dataset(fresh_valid)
    # Repeat known cases so a minibatch sees them frequently, but retain a hard
    # exact known gate below rather than trusting this weighting.
    train_x = np.concatenate([fresh_x, np.tile(known_x, (12, 1, 1))])
    train_y = np.concatenate([fresh_y, np.tile(known_y, (12, 1))])
    tensor_x = torch.from_numpy(train_x)
    tensor_y = torch.from_numpy(train_y)

    source_model = onnx.load(SOURCE)
    source_weights = next(
        numpy_helper.to_array(item).astype(np.float32).reshape(-1)
        for item in source_model.graph.initializer
        if item.name == "score_W_q"
    )
    rng = np.random.default_rng(args.seed)
    trials: list[dict[str, object]] = []
    best: tuple[int, np.ndarray, int] | None = None

    for restart in range(args.restarts):
        if restart == 0:
            initial = source_weights / max(float(np.linalg.norm(source_weights)), 1.0)
        else:
            initial = source_weights / max(float(np.linalg.norm(source_weights)), 1.0)
            initial += rng.normal(0.0, 0.12 + 0.03 * restart, size=36).astype(np.float32)
        weights = torch.nn.Parameter(torch.from_numpy(initial.copy()))
        optimizer = torch.optim.Adam([weights], lr=0.025)
        generator = torch.Generator().manual_seed(args.seed + restart)
        for _epoch in range(args.epochs):
            order = torch.randperm(len(tensor_x), generator=generator)
            for start in range(0, len(order), 768):
                indices = order[start : start + 768]
                patches = tensor_x[indices]
                targets = tensor_y[indices]
                scores = torch.einsum("bif,f->bi", patches, weights)
                positive = scores.masked_fill(~targets, float("inf")).amin(dim=1)
                negative = scores.masked_fill(targets, float("-inf"))
                # Smooth maximum supplies gradients to several plausible false
                # anchors, while the minimum enforces retention of every box.
                false_peak = torch.logsumexp(negative * 2.0, dim=1) / 2.0
                loss = functional.softplus(false_peak - positive + 0.20).mean()
                loss = loss + 1.0e-4 * weights.square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        floating = weights.detach().numpy()
        for quantized in quantizations(floating):
            known_right, known_total = exact_rate(known_x, known_y, quantized)
            if known_right != known_total:
                continue
            valid_right, valid_total = exact_rate(valid_x, valid_y, quantized)
            trials.append(
                {
                    "restart": restart,
                    "valid_right_numpy": valid_right,
                    "valid_total": valid_total,
                    "known_right_numpy": known_right,
                    "weights": quantized.astype(int).tolist(),
                }
            )
            if best is None or valid_right > best[0]:
                best = (valid_right, quantized.copy(), restart)

    if best is None:
        raise RuntimeError("no quantization preserved all known examples")

    trials.sort(key=lambda row: int(row["valid_right_numpy"]), reverse=True)
    # ORT-confirm the best distinct candidates because its uint8 quantization
    # and TopK tie policy, rather than the numpy approximation, is authoritative.
    finalists = []
    seen: set[bytes] = set()
    for row in trials:
        value = np.asarray(row["weights"], dtype=np.int8)
        if value.tobytes() in seen:
            continue
        seen.add(value.tobytes())
        model = copy.deepcopy(source_model)
        replace_weights(model, value)
        known_disabled = ort_rate(model, known, "disabled")
        known_default = ort_rate(model, known, "default")
        valid_disabled = ort_rate(model, fresh_valid, "disabled")
        valid_default = ort_rate(model, fresh_valid, "default")
        candidate_path = HERE / f"task023_ranker_{len(finalists):02d}.onnx"
        onnx.save(model, candidate_path)
        finalists.append(
            {
                **row,
                "path": str(candidate_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                "known_disabled": known_disabled,
                "known_default": known_default,
                "valid_disabled": valid_disabled,
                "valid_default": valid_default,
            }
        )
        if len(finalists) >= 12:
            break

    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "train_count": args.train,
        "valid_count": args.valid,
        "seed": args.seed,
        "generator_non_injective": True,
        "finalists": finalists,
    }
    (HERE / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    winner = finalists[0]
    return 0 if min(winner["known_disabled"][0], winner["known_default"][0]) == len(known) else 2


if __name__ == "__main__":
    raise SystemExit(main())
