#!/usr/bin/env python3
"""Train the cost-1614 two-stage QLinearConv ranker proposed for task023.

The lane is empirical POLICY90 work.  It preserves the clean output assembly,
requires every stored example, and writes only local candidates.
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
import torch.nn.functional as F
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
KNOWN_PATH = ROOT / "inputs/neurogolf-2026/task023.json"
SOURCE = ROOT / (
    "scripts/golf/loop_8004_42_plus20/agent_clean95_all/candidates/"
    "task023_9a2b78138891_cost1541.onnx"
)
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def onehot(grid: list[list[int]]) -> np.ndarray:
    values = np.asarray(grid, dtype=np.int64)
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    rows, cols = np.indices(values.shape)
    result[0, values, rows, cols] = 1.0
    return result


def generated_cases(count: int, seed: int) -> list[dict]:
    common = importlib.import_module("common")
    generator = importlib.import_module("task_150deff5")
    result = []
    for index in range(count):
        current = seed + index
        random.seed(current)
        common.random.seed(current)
        result.append(generator.generate())
    return result


def known_cases() -> list[dict]:
    data = json.loads(KNOWN_PATH.read_text())
    return [item for split in ("train", "test", "arc-gen") for item in data.get(split, [])]


def case_features(example: dict) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(example["input"], dtype=np.int8)
    output = np.asarray(example["output"], dtype=np.int8)
    gray = (grid[:8, 1:9] == 5).astype(np.float32)
    padded = np.pad(gray, ((1, 0), (1, 0)))
    patches = np.stack([
        padded[row : row + 4, col : col + 4].reshape(-1)
        for row in range(6) for col in range(6)
    ])
    cyan = output == 8
    target = np.zeros(36, dtype=bool)
    for row in range(output.shape[0] - 1):
        for col in range(output.shape[1] - 1):
            if bool(cyan[row : row + 2, col : col + 2].all()):
                score_col = col - 1
                if 0 <= row < 6 and 0 <= score_col < 6:
                    target[row * 6 + score_col] = True
    expected = 2 if grid.shape[1] == 9 else 3
    if int(target.sum()) != expected:
        raise ValueError(f"unexpected target count {target.sum()} != {expected}")
    return patches, target


def dataset(cases: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rows = [case_features(case) for case in cases]
    return np.stack([row[0] for row in rows]), np.stack([row[1] for row in rows])


def exact_rate(
    features: np.ndarray,
    targets: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: int,
) -> tuple[int, int]:
    hidden = np.einsum(
        "nif,rf->nir", features.astype(np.int32), w1.astype(np.int32)
    ) + b1.astype(np.int32)
    hidden = np.clip(hidden, 0, 255)
    raw = np.einsum("nir,r->ni", hidden, w2.astype(np.int32)) + int(b2)
    scores = np.clip(raw, 0, 255)
    right = 0
    for score, target in zip(scores, targets):
        count = int(target.sum())
        selected = np.argsort(-score, kind="stable")[:count]
        right += bool(target[selected].all())
    return right, len(features)


def quantizations(
    w1f: np.ndarray,
    b1f: np.ndarray,
    w2f: np.ndarray,
    b2f: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    max1 = max(float(np.abs(w1f).max()), 1.0e-6)
    max2 = max(float(np.abs(w2f).max()), 1.0e-6)
    result = []
    seen = set()
    for peak1 in (4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 112, 127):
        scale1 = peak1 / max1
        w1 = np.clip(np.rint(w1f * scale1), -127, 127).astype(np.int8)
        b1 = np.clip(np.rint(b1f * scale1), -(2**31), 2**31 - 1).astype(np.int32)
        for peak2 in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 127):
            scale2 = peak2 / max2
            w2 = np.clip(np.rint(w2f * scale2), -127, 127).astype(np.int8)
            b2 = int(np.clip(np.rint(b2f * scale1 * scale2), -(2**31), 2**31 - 1))
            key = w1.tobytes() + b1.tobytes() + w2.tobytes() + np.int32(b2).tobytes()
            if key not in seen:
                seen.add(key)
                result.append((w1, b1, w2, b2))
    return result


def build_model(
    source: onnx.ModelProto,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: int,
) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    nodes = list(model.graph.node)
    target = next(index for index, node in enumerate(nodes) if node.op_type == "QLinearConv")
    old = nodes[target]
    if list(old.output) != ["score6"]:
        raise RuntimeError("unexpected source QLinearConv")
    first = helper.make_node(
        "QLinearConv",
        ["gray8_u8", "x_scale", "x_zero", "morph_w1", "w_scale", "w_zero",
         "y_scale", "y_zero", "morph_b1"],
        ["score6_hidden"],
        name="morphology_stage1",
        pads=[1, 1, 0, 0],
    )
    second = helper.make_node(
        "QLinearConv",
        ["score6_hidden", "y_scale", "y_zero", "morph_w2", "w_scale", "w_zero",
         "y_scale", "y_zero", "morph_b2"],
        ["score6"],
        name="morphology_stage2",
    )
    del model.graph.node[:]
    model.graph.node.extend(nodes[:target] + [first, second] + nodes[target + 1 :])
    kept = [item for item in model.graph.initializer if item.name != "score_W_q"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend([
        numpy_helper.from_array(w1.reshape(2, 1, 4, 4), "morph_w1"),
        numpy_helper.from_array(b1.reshape(2), "morph_b1"),
        numpy_helper.from_array(w2.reshape(1, 2, 1, 1), "morph_w2"),
        numpy_helper.from_array(np.asarray([b2], dtype=np.int32), "morph_b2"),
    ])
    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def ort_rate(model: onnx.ModelProto, cases: list[dict], disabled: bool) -> tuple[int, int, int]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    right = errors = 0
    for example in cases:
        try:
            raw = session.run(["output"], {"input": onehot(example["input"])})[0]
            right += bool(np.array_equal(raw > 0, onehot(example["output"]).astype(bool)))
        except Exception:  # noqa: BLE001
            errors += 1
    return right, len(cases), errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=40000)
    parser.add_argument("--valid", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=245023001)
    args = parser.parse_args()
    torch.set_num_threads(4)

    known = known_cases()
    train_cases = generated_cases(args.train, args.seed)
    valid_cases = generated_cases(args.valid, args.seed + 10_000_000)
    known_x, known_y = dataset(known)
    train_x, train_y = dataset(train_cases)
    valid_x, valid_y = dataset(valid_cases)
    fit_x = np.concatenate([train_x, np.tile(known_x, (20, 1, 1))])
    fit_y = np.concatenate([train_y, np.tile(known_y, (20, 1))])
    tx = torch.from_numpy(fit_x)
    ty = torch.from_numpy(fit_y)
    kx = torch.from_numpy(known_x)
    ky = torch.from_numpy(known_y)
    rng = np.random.default_rng(args.seed)
    source = onnx.load(SOURCE)
    finalists = []
    incomplete = []
    seen = set()

    for restart in range(args.restarts):
        w1 = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.25, (2, 16)).astype(np.float32)))
        b1 = torch.nn.Parameter(torch.from_numpy(rng.normal(0, 0.10, 2).astype(np.float32)))
        w2 = torch.nn.Parameter(torch.from_numpy(np.asarray([1.0, 1.0], dtype=np.float32)))
        b2 = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=0.025)
        generator = torch.Generator().manual_seed(args.seed + restart)
        for _epoch in range(args.epochs):
            order = torch.randperm(len(tx), generator=generator)
            for start in range(0, len(order), 1024):
                indices = order[start : start + 1024]
                hidden = torch.clamp(torch.einsum("bif,rf->bir", tx[indices], w1) + b1, 0, 16)
                scores = torch.einsum("bir,r->bi", hidden, w2) + b2
                targets = ty[indices]
                positive = scores.masked_fill(~targets, float("inf")).amin(dim=1)
                negative = scores.masked_fill(targets, float("-inf"))
                false_peak = torch.logsumexp(negative * 2.0, dim=1) / 2.0
                loss = F.softplus(false_peak - positive + 0.25).mean()
                known_hidden = torch.clamp(torch.einsum("bif,rf->bir", kx, w1) + b1, 0, 16)
                known_scores = torch.einsum("bir,r->bi", known_hidden, w2) + b2
                known_positive = known_scores.masked_fill(~ky, float("inf")).amin(dim=1)
                known_negative = known_scores.masked_fill(ky, float("-inf"))
                known_false_peak = torch.logsumexp(known_negative * 2.0, dim=1) / 2.0
                loss = loss + 4.0 * F.softplus(
                    known_false_peak - known_positive + 0.35
                ).mean()
                loss = loss + 1.0e-5 * (w1.square().mean() + w2.square().mean())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        arrays = (w1.detach().numpy(), b1.detach().numpy(), w2.detach().numpy(), float(b2.detach()))
        for q in quantizations(*arrays):
            key = b"".join(np.asarray(value).tobytes() for value in q)
            if key in seen:
                continue
            seen.add(key)
            known_right, known_total = exact_rate(known_x, known_y, *q)
            valid_right, valid_total = exact_rate(valid_x, valid_y, *q)
            trial = {
                "restart": restart,
                "known_right_numpy": known_right,
                "known_total": known_total,
                "valid_right_numpy": valid_right,
                "valid_total": valid_total,
                "w1": q[0].astype(int).tolist(),
                "b1": q[1].astype(int).tolist(),
                "w2": q[2].astype(int).tolist(),
                "b2": q[3],
            }
            if known_right == known_total:
                finalists.append(trial)
            else:
                incomplete.append(trial)

    finalists.sort(key=lambda row: row["valid_right_numpy"], reverse=True)
    audited = []
    for index, row in enumerate(finalists[:20]):
        q = (
            np.asarray(row["w1"], dtype=np.int8),
            np.asarray(row["b1"], dtype=np.int32),
            np.asarray(row["w2"], dtype=np.int8),
            int(row["b2"]),
        )
        model = build_model(source, *q)
        path = HERE / f"task023_morphology_{index:02d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
        known_disabled = ort_rate(model, known, True)
        known_default = ort_rate(model, known, False)
        valid_disabled = ort_rate(model, valid_cases, True)
        valid_default = ort_rate(model, valid_cases, False)
        audited.append({
            **row,
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "memory": int(memory),
            "params": int(params),
            "cost": int(cost),
            "known_disabled": known_disabled,
            "known_default": known_default,
            "valid_disabled": valid_disabled,
            "valid_default": valid_default,
        })

    report = {
        "authority_cost": 1622,
        "source": str(SOURCE.relative_to(ROOT)),
        "train_count": args.train,
        "valid_count": args.valid,
        "restarts": args.restarts,
        "epochs": args.epochs,
        "seed": args.seed,
        "quantized_known_complete_count": len(finalists),
        "best_incomplete": sorted(
            incomplete,
            key=lambda row: (row["known_right_numpy"], row["valid_right_numpy"]),
            reverse=True,
        )[:20],
        "audited": audited,
    }
    (HERE / "search.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "quantized_known_complete_count": len(finalists),
        "best_incomplete": report["best_incomplete"][:3],
        "best": audited[0] if audited else None,
    }, indent=2))
    return 0 if audited else 2


if __name__ == "__main__":
    raise SystemExit(main())
