#!/usr/bin/env python3
"""Known-constrained search for task023's cost-1621 spatial morphology ranker.

The clean cost-1541 graph has one 6x6 QLinearConv.  This lane replaces it with
a bias-free 4x4x2 QLinearConv and a bias-free spatial QLinearConv.  Both 2x3
and 3x2 second-stage kernels and every padding split that retains 6x6 hidden
and score maps are searched.  The unchanged tail still performs TopK and
paints cyan boxes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
KNOWN_PATH = ROOT / "inputs/neurogolf-2026/task023.json"
SOURCE = ROOT / (
    "scripts/golf/loop_8004_42_plus20/root_task023_tune80/"
    "task023_ranker_coordinate2.onnx"
)
sys.path.insert(0, str(TASK_DIR))


@dataclass(frozen=True)
class Layout:
    kind: str
    kernel2: tuple[int, int]
    pads1: tuple[int, int, int, int]
    pads2: tuple[int, int, int, int]

    @property
    def label(self) -> str:
        return (
            f"{self.kind}_p1_{''.join(map(str, self.pads1))}_"
            f"p2_{''.join(map(str, self.pads2))}"
        )


def layouts() -> list[Layout]:
    """Enumerate all 48 shape-preserving padding orientations."""
    result: list[Layout] = []
    for kind, (kh, kw) in (("A2x3", (2, 3)), ("B3x2", (3, 2))):
        for top1 in range(2):
            for left1 in range(2):
                pads1 = (top1, left1, 1 - top1, 1 - left1)
                for top2 in range(kh):
                    for left2 in range(kw):
                        pads2 = (top2, left2, kh - 1 - top2, kw - 1 - left2)
                        result.append(Layout(kind, (kh, kw), pads1, pads2))
    if len(result) != 48 or len({item.label for item in result}) != 48:
        raise AssertionError("padding enumeration must contain 48 unique layouts")
    return result


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


def case_arrays(example: dict) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(example["input"], dtype=np.int8)
    output = np.asarray(example["output"], dtype=np.int8)
    gray = (grid[:8, 1:9] == 5).astype(np.float32)
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
    return gray, target


def dataset(cases: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rows = [case_arrays(case) for case in cases]
    return np.stack([row[0] for row in rows]), np.stack([row[1] for row in rows])


def torch_scores(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    layout: Layout,
) -> torch.Tensor:
    top1, left1, bottom1, right1 = layout.pads1
    top2, left2, bottom2, right2 = layout.pads2
    padded = F.pad(x[:, None], (left1, right1, top1, bottom1))
    hidden = torch.clamp(F.conv2d(padded, w1), 0.0, 255.0)
    hidden = F.pad(hidden, (left2, right2, top2, bottom2))
    scores = F.conv2d(hidden, w2)
    return scores.flatten(1)


def ranking_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positive_floor = scores.masked_fill(~targets, float("inf")).amin(dim=1)
    negative = scores.masked_fill(targets, float("-inf"))
    false_peak = torch.logsumexp(negative * 2.0, dim=1) / 2.0
    return F.softplus(false_peak - positive_floor + 0.25).mean()


def teacher_scores(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    padded = F.pad(x[:, None], (2, 1, 2, 1))
    return torch.clamp(F.conv2d(padded, weights), 0.0, 255.0).flatten(1)


def stage1_patches(x: np.ndarray, layout: Layout) -> np.ndarray:
    top, left, bottom, right = layout.pads1
    padded = np.pad(x, ((0, 0), (top, bottom), (left, right)))
    view = np.lib.stride_tricks.sliding_window_view(padded, (4, 4), axis=(1, 2))
    if view.shape[1:3] != (6, 6):
        raise AssertionError(view.shape)
    return np.ascontiguousarray(view.reshape(len(x), 36, 16), dtype=np.int16)


def second_patches(hidden: np.ndarray, layout: Layout) -> np.ndarray:
    top, left, bottom, right = layout.pads2
    padded = np.pad(hidden, ((0, 0), (0, 0), (top, bottom), (left, right)))
    view = np.lib.stride_tricks.sliding_window_view(
        padded, layout.kernel2, axis=(2, 3)
    )
    if view.shape[2:4] != (6, 6):
        raise AssertionError(view.shape)
    return np.ascontiguousarray(view, dtype=np.int16)


def integer_scores(
    patches1: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    layout: Layout,
) -> np.ndarray:
    w1 = w1.reshape(2, 16)
    pre = np.einsum(
        "nif,cf->nci", patches1.astype(np.int32), w1.astype(np.int32), optimize=True
    )
    hidden = np.clip(pre, 0, 255).reshape(len(patches1), 2, 6, 6).astype(np.int16)
    patches2 = second_patches(hidden, layout)
    raw = np.einsum(
        "ncrsij,cij->nrs",
        patches2.astype(np.int32),
        w2.reshape(2, *layout.kernel2).astype(np.int32),
        optimize=True,
    )
    return np.clip(raw, 0, 255).reshape(len(patches1), 36).astype(np.int16)


def success_mask(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Match the clean graph's TopK ordering, including deterministic ties."""
    keys = scores.astype(np.int32) * 64 - np.arange(36, dtype=np.int32)[None, :]
    positive_floor = np.where(targets, keys, np.iinfo(np.int32).max).min(axis=1)
    negative_peak = np.where(targets, np.iinfo(np.int32).min, keys).max(axis=1)
    return positive_floor > negative_peak


def quantizations(w1f: np.ndarray, w2f: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    max1 = max(float(np.abs(w1f).max()), 1.0e-7)
    max2 = max(float(np.abs(w2f).max()), 1.0e-7)
    peaks1 = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
    peaks2 = (1, 2, 3, 4, 6, 8, 12, 16, 24)
    result = []
    seen = set()
    for peak1 in peaks1:
        q1 = np.clip(np.rint(w1f * (peak1 / max1)), -127, 127).astype(np.int8)
        for peak2 in peaks2:
            q2 = np.clip(np.rint(w2f * (peak2 / max2)), -127, 127).astype(np.int8)
            key = q1.tobytes() + q2.tobytes()
            if key not in seen and bool(q1.any()) and bool(q2.any()):
                seen.add(key)
                result.append((q1, q2))
    return result


def build_model(
    source: onnx.ModelProto,
    layout: Layout,
    w1: np.ndarray,
    w2: np.ndarray,
) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    nodes = list(model.graph.node)
    target = next(index for index, node in enumerate(nodes) if node.op_type == "QLinearConv")
    old = nodes[target]
    if list(old.input)[:4] != ["gray8_u8", "x_scale", "x_zero", "score_W_q"]:
        raise RuntimeError("unexpected source QLinearConv")
    first = helper.make_node(
        "QLinearConv",
        ["gray8_u8", "x_scale", "x_zero", "morph_w1", "w_scale", "w_zero",
         "y_scale", "y_zero"],
        ["score6_hidden"],
        name="spatial_morphology_stage1",
        pads=list(layout.pads1),
    )
    second = helper.make_node(
        "QLinearConv",
        ["score6_hidden", "y_scale", "y_zero", "morph_w2", "w_scale", "w_zero",
         "y_scale", "y_zero"],
        ["score6"],
        name="spatial_morphology_stage2",
        pads=list(layout.pads2),
    )
    del model.graph.node[:]
    model.graph.node.extend(nodes[:target] + [first, second] + nodes[target + 1 :])
    kept = [item for item in model.graph.initializer if item.name != "score_W_q"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(w1.astype(np.int8).reshape(2, 1, 4, 4), "morph_w1"),
            numpy_helper.from_array(
                w2.astype(np.int8).reshape(1, 2, *layout.kernel2), "morph_w2"
            ),
        ]
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
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


def train_once(
    layout: Layout,
    train_x: np.ndarray,
    train_y: np.ndarray,
    known_x: np.ndarray,
    known_y: np.ndarray,
    *,
    epochs: int,
    seed: int,
    known_weight: float,
    teacher_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    w1 = torch.nn.Parameter(torch.from_numpy(rng.normal(0.0, 0.20, (2, 1, 4, 4)).astype(np.float32)))
    # Positive output weights avoid the all-zero clipped QLinearConv basin.
    w2 = torch.nn.Parameter(
        torch.from_numpy(
            rng.normal(0.35, 0.15, (1, 2, *layout.kernel2)).astype(np.float32)
        )
    )
    optimizer = torch.optim.Adam([w1, w2], lr=0.025)
    tx = torch.from_numpy(train_x)
    ty = torch.from_numpy(train_y)
    kx = torch.from_numpy(known_x)
    ky = torch.from_numpy(known_y)
    source_model = onnx.load(SOURCE)
    source_w = next(
        numpy_helper.to_array(item).astype(np.float32)
        for item in source_model.graph.initializer
        if item.name == "score_W_q"
    )
    teacher_w = torch.from_numpy(source_w)
    train_teacher = teacher_scores(tx, teacher_w) if teacher_weight else None
    known_teacher = teacher_scores(kx, teacher_w) if teacher_weight else None
    last_loss = 0.0
    for _epoch in range(epochs):
        order = torch.randperm(len(tx), generator=generator)
        for start in range(0, len(order), 768):
            indices = order[start : start + 768]
            score = torch_scores(tx[indices], w1, w2, layout)
            loss = ranking_loss(score, ty[indices])
            known_score = torch_scores(kx, w1, w2, layout)
            loss = loss + known_weight * ranking_loss(known_score, ky)
            if teacher_weight:
                loss = loss + teacher_weight * F.mse_loss(
                    torch.clamp(score, 0.0, 255.0), train_teacher[indices]
                ) / 4096.0
                loss = loss + teacher_weight * F.mse_loss(
                    torch.clamp(known_score, 0.0, 255.0), known_teacher
                ) / 4096.0
            loss = loss + 2.0e-5 * (w1.square().mean() + w2.square().mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach())
    with torch.no_grad():
        known_float = success_mask(
            np.clip(torch_scores(kx, w1, w2, layout).numpy(), 0, 255).astype(np.int16),
            known_y,
        )
    return (
        w1.detach().numpy(),
        w2.detach().numpy(),
        {"last_loss": last_loss, "known_float_right": int(known_float.sum())},
    )


def evaluate_quantizations(
    layout: Layout,
    w1f: np.ndarray,
    w2f: np.ndarray,
    known_patches: np.ndarray,
    known_y: np.ndarray,
    valid_patches: np.ndarray,
    valid_y: np.ndarray,
    *,
    restart: int,
    training: dict[str, float],
) -> list[dict]:
    trials = []
    for w1, w2 in quantizations(w1f, w2f):
        known_score = integer_scores(known_patches, w1, w2, layout)
        known_right = int(success_mask(known_score, known_y).sum())
        valid_score = integer_scores(valid_patches, w1, w2, layout)
        valid_right = int(success_mask(valid_score, valid_y).sum())
        trials.append(
            {
                "layout": layout.label,
                "restart": restart,
                **training,
                "known_right": known_right,
                "known_total": len(known_y),
                "valid_right": valid_right,
                "valid_total": len(valid_y),
                "w1": w1.astype(int).reshape(2, 16).tolist(),
                "w2": w2.astype(int).reshape(2, -1).tolist(),
            }
        )
    return trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=12000)
    parser.add_argument("--valid", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=246023001)
    parser.add_argument("--known-weight", type=float, default=2.0)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--output", type=Path, default=HERE / "screen.json")
    parser.add_argument("--save-best", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)

    selected = layouts()
    if args.only:
        selected = [item for item in selected if item.label in set(args.only)]
        missing = sorted(set(args.only) - {item.label for item in selected})
        if missing:
            raise ValueError(f"unknown layouts: {missing}")

    known = known_cases()
    train_cases = generated_cases(args.train, args.seed)
    valid_cases = generated_cases(args.valid, args.seed + 10_000_000)
    known_x, known_y = dataset(known)
    train_x, train_y = dataset(train_cases)
    valid_x, valid_y = dataset(valid_cases)
    records = []
    all_trials = []
    for layout_index, layout in enumerate(selected):
        known_patches = stage1_patches(known_x, layout)
        valid_patches = stage1_patches(valid_x, layout)
        candidates = []
        for restart in range(args.restarts):
            w1f, w2f, training = train_once(
                layout,
                train_x,
                train_y,
                known_x,
                known_y,
                epochs=args.epochs,
                seed=args.seed + layout_index * 1009 + restart,
                known_weight=args.known_weight,
                teacher_weight=args.teacher_weight,
            )
            candidates.extend(
                evaluate_quantizations(
                    layout,
                    w1f,
                    w2f,
                    known_patches,
                    known_y,
                    valid_patches,
                    valid_y,
                    restart=restart,
                    training=training,
                )
            )
        candidates.sort(
            key=lambda row: (row["known_right"], row["valid_right"]), reverse=True
        )
        known_complete = [row for row in candidates if row["known_right"] == len(known_y)]
        known_complete.sort(key=lambda row: row["valid_right"], reverse=True)
        best_valid = max(candidates, key=lambda row: row["valid_right"])
        record = {
            "layout": asdict(layout),
            "label": layout.label,
            "trial_count": len(candidates),
            "best_lexicographic": candidates[0],
            "best_valid": best_valid,
            "best_known_complete": known_complete[0] if known_complete else None,
        }
        records.append(record)
        all_trials.extend(candidates[:3])
        all_trials.extend(known_complete[:3])
        print(
            json.dumps(
                {
                    "progress": [layout_index + 1, len(selected)],
                    "layout": layout.label,
                    "best_known": candidates[0]["known_right"],
                    "best_valid_at_known": candidates[0]["valid_right"],
                    "best_valid": best_valid["valid_right"],
                    "known_complete": len(known_complete),
                }
            ),
            flush=True,
        )

    complete = [
        row
        for record in records
        for row in ([record["best_known_complete"]] if record["best_known_complete"] else [])
    ]
    complete.sort(key=lambda row: row["valid_right"], reverse=True)
    best = complete[0] if complete else max(
        (record["best_lexicographic"] for record in records),
        key=lambda row: (row["known_right"], row["valid_right"]),
    )
    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "seed": args.seed,
        "train_count": args.train,
        "valid_count": args.valid,
        "epochs": args.epochs,
        "restarts": args.restarts,
        "known_weight": args.known_weight,
        "teacher_weight": args.teacher_weight,
        "orientation_count": len(selected),
        "all_orientation_count": len(layouts()),
        "records": records,
        "best": best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.save_best:
        layout = next(item for item in layouts() if item.label == best["layout"])
        model = build_model(
            onnx.load(SOURCE),
            layout,
            np.asarray(best["w1"], dtype=np.int8),
            np.asarray(best["w2"], dtype=np.int8),
        )
        onnx.save(model, args.save_best)
        print(
            json.dumps(
                {
                    "saved": str(args.save_best),
                    "sha256": hashlib.sha256(args.save_best.read_bytes()).hexdigest(),
                    "known_disabled": ort_rate(model, known, True),
                    "known_default": ort_rate(model, known, False),
                }
            )
        )
    print(json.dumps({"best": best, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
