#!/usr/bin/env python3
"""Train color-equivariant soft local task192 classifiers.

Unlike the earlier per-color hard LP, this model ties coefficients according
to their semantic role: background, selected box color, and every other
nonzero color.  The final ONNX is still the cheap histogram + dynamic-bias +
single-Conv graph.  No examples or outputs are serialized into the model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))


KERNELS = {
    "k3": (3, 3, (1, 1, 1, 1)),
    "k4l": (3, 4, (1, 2, 1, 1)),
    "k4r": (3, 4, (1, 1, 1, 2)),
    "k4t": (4, 3, (2, 1, 1, 1)),
    "k4b": (4, 3, (1, 1, 2, 1)),
    "k5": (3, 5, (1, 2, 1, 2)),
}


def selected_color(grid: list[list[int]]) -> int:
    flat = sum(grid, [])
    return max(range(1, 10), key=flat.count)


def expected_grid(example: dict[str, object]) -> np.ndarray:
    grid = example["output"]
    assert isinstance(grid, list)
    result = np.full((30, 30), -1, dtype=np.int8)
    height, width = len(grid), len(grid[0])
    result[:height, :width] = np.asarray(grid, dtype=np.int8)
    return result


def feature_planes(
    example: dict[str, object], kernel: str, background_features: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kh, kw, pads = KERNELS[kernel]
    top, left, bottom, right = pads
    grid = np.asarray(example["input"], dtype=np.int8)
    height, width = grid.shape
    onehot = np.zeros((10, 30, 30), dtype=np.uint8)
    for color in range(10):
        onehot[color, :height, :width] = grid == color
    padded = np.pad(onehot, ((0, 0), (top, bottom), (left, right)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(1, 2))
    # [C,30,30,kh,kw] -> [900,C,kh,kw]
    windows = windows.transpose(1, 2, 0, 3, 4).reshape(900, 10, kh, kw)
    selected = selected_color(example["input"])
    nonzero = windows[:, 1:].sum(axis=1, dtype=np.uint8)
    if background_features == "full":
        x0 = windows.reshape(900, -1)
    else:
        x0 = np.stack((windows[:, 0], nonzero), axis=1).reshape(900, -1)
    own = windows[:, selected]
    other = nonzero - own
    xs = np.stack((windows[:, 0], own, other), axis=1).reshape(900, -1)
    expected = expected_grid(example).reshape(-1)
    y0 = expected == 0
    ys = expected == selected
    return x0, y0, xs, ys


def add_records(
    table: dict[bytes, list[int]], features: np.ndarray, labels: np.ndarray, known: bool
) -> None:
    for feature, positive in zip(features, labels, strict=True):
        counts = table[feature.tobytes()]
        offset = 2 if known else 0
        counts[offset + (0 if positive else 1)] += 1


def to_arrays(table: dict[bytes, list[int]], dimension: int) -> tuple[torch.Tensor, torch.Tensor]:
    features = np.asarray(
        [np.frombuffer(key, dtype=np.uint8, count=dimension) for key in table],
        dtype=np.float32,
    )
    counts = np.asarray(list(table.values()), dtype=np.float32)
    return torch.from_numpy(features), torch.from_numpy(counts)


def fit_classifier(
    features: torch.Tensor,
    counts: torch.Tensor,
    count_power: float,
    known_weight: float,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, float, dict[str, object]]:
    torch.manual_seed(seed)
    dimension = features.shape[1]
    weight = torch.nn.Parameter(torch.zeros(dimension, dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    x = features.to(torch.float64)
    c = counts.to(torch.float64)
    # Each unique patch/label pair gets a base vote; count_power controls how
    # strongly common patches dominate rare edge/corner patterns.
    fresh_pos = torch.where(c[:, 0] > 0, c[:, 0].pow(count_power), c[:, 0])
    fresh_neg = torch.where(c[:, 1] > 0, c[:, 1].pow(count_power), c[:, 1])
    known_pos = (c[:, 2] > 0).to(torch.float64) * known_weight
    known_neg = (c[:, 3] > 0).to(torch.float64) * known_weight
    pos_vote = fresh_pos + known_pos
    neg_vote = fresh_neg + known_neg
    normalizer = (pos_vote + neg_vote).sum()
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=500,
        max_eval=650,
        tolerance_grad=1e-11,
        tolerance_change=1e-13,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        score = x @ weight + bias
        loss = (
            pos_vote * F.softplus(-score) + neg_vote * F.softplus(score)
        ).sum() / normalizer
        loss = loss + l2 * weight.square().mean()
        loss.backward()
        return loss

    final_loss = float(optimizer.step(closure).detach())
    with torch.no_grad():
        score = x @ weight + bias
        known_bad = int(
            (((c[:, 2] > 0) & (score <= 0)) | ((c[:, 3] > 0) & (score > 0))).sum()
        )
        fresh_vote_errors = float(
            (
                c[:, 0] * (score <= 0).to(torch.float64)
                + c[:, 1] * (score > 0).to(torch.float64)
            ).sum()
        )
        report = {
            "unique_patches": int(len(features)),
            "dimension": int(dimension),
            "loss": final_loss,
            "known_bad_unique_patch_labels": known_bad,
            "fresh_weighted_cell_errors": fresh_vote_errors,
            "weight_l2": float(weight.norm()),
            "bias": float(bias),
        }
    return weight.detach().numpy().astype(np.float32), float(bias.detach()), report


def make_model(
    kernel: str,
    background_features: str,
    background_weight: np.ndarray,
    background_bias: float,
    selected_weight: np.ndarray,
    selected_bias: float,
    scale: float,
) -> onnx.ModelProto:
    kh, kw, pads = KERNELS[kernel]
    background_channels = 10 if background_features == "full" else 2
    w0 = background_weight.reshape(background_channels, kh, kw)
    ws = selected_weight.reshape(3, kh, kw)
    weights = np.zeros((10, 10, kh, kw), dtype=np.float32)
    if background_features == "full":
        weights[0] = w0
    else:
        weights[0, 0] = w0[0]
        weights[0, 1:] = w0[1]
    for output_color in range(1, 10):
        weights[output_color, 0] = ws[0]
        weights[output_color, output_color] = ws[1]
        for input_color in range(1, 10):
            if input_color != output_color:
                weights[output_color, input_color] = ws[2]
    weights *= np.float32(scale)
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    initializers = [
        numpy_helper.from_array(np.asarray([0] + [1] * 9, np.float32), "sel_nz"),
        numpy_helper.from_array(
            np.asarray([background_bias * scale] + [-1_000_000.0] * 9, np.float32),
            "bias_base",
        ),
        numpy_helper.from_array(np.asarray([selected_bias * scale], np.float32), "bias_update"),
        numpy_helper.from_array(weights, "W"),
    ]
    nodes = [
        helper.make_node("Einsum", ["input", "sel_nz"], ["hist"], equation="bchw,c->c"),
        helper.make_node("ArgMax", ["hist"], ["selected"], axis=0, keepdims=1),
        helper.make_node(
            "ScatterElements", ["bias_base", "selected", "bias_update"], ["bias"], axis=0
        ),
        helper.make_node("Conv", ["input", "W", "bias"], ["output"], pads=list(pads)),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, f"task192_soft_symmetric_{kernel}", [input_info], [output_info], initializers),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def tensorize(example: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(example["input"], dtype=np.int8)
    expected = expected_grid(example)
    height, width = grid.shape
    input_tensor = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for color in range(10):
        input_tensor[0, color, :height, :width] = grid == color
    expected_tensor = np.zeros((1, 10, 30, 30), dtype=np.bool_)
    for color in range(10):
        expected_tensor[0, color] = expected == color
    return input_tensor, expected_tensor


def evaluate(model: onnx.ModelProto, examples: list[dict[str, object]]) -> dict[str, object]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    inference = ort.InferenceSession(model.SerializeToString(), options)
    right = wrong = errors = differing = 0
    min_true = math.inf
    max_false = -math.inf
    for example in examples:
        input_tensor, expected = tensorize(example)
        try:
            raw = inference.run(None, {"input": input_tensor})[0]
        except Exception:  # noqa: BLE001
            errors += 1
            continue
        diff = int(np.count_nonzero((raw > 0) != expected))
        right += int(diff == 0)
        wrong += int(diff != 0)
        differing += diff
        min_true = min(min_true, float(raw[expected].min()))
        max_false = max(max_false, float(raw[~expected].max()))
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "total": len(examples),
        "accuracy": right / len(examples),
        "differing_elements": differing,
        "min_true": min_true,
        "max_false": max_false,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=tuple(KERNELS), default="k4l")
    parser.add_argument(
        "--background-features", choices=("symmetric", "full"), default="full"
    )
    parser.add_argument("--fresh-train", type=int, default=500)
    parser.add_argument("--seeds", default="79192101,79192102")
    parser.add_argument("--count-powers", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--known-weight", type=float, default=10000.0)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--scale", type=float, default=128.0)
    parser.add_argument("--prefix", default="soft")
    args = parser.parse_args()

    known_data = json.loads((ROOT / "inputs/neurogolf-2026/task192.json").read_text())
    known = known_data["train"] + known_data["test"] + known_data["arc-gen"]
    generator = importlib.import_module("task_7e0986d6")
    fresh = []
    seed_values = [int(value) for value in args.seeds.split(",")]
    per_seed = math.ceil(args.fresh_train / len(seed_values))
    for seed in seed_values:
        random.seed(seed)
        for _ in range(per_seed):
            fresh.append(generator.generate())
    fresh = fresh[: args.fresh_train]
    print(f"generated fresh={len(fresh)} seeds={seed_values}", flush=True)

    background = defaultdict(lambda: [0, 0, 0, 0])
    selected = defaultdict(lambda: [0, 0, 0, 0])
    for is_known, examples in ((True, known), (False, fresh)):
        for index, example in enumerate(examples):
            x0, y0, xs, ys = feature_planes(
                example, args.kernel, args.background_features
            )
            add_records(background, x0, y0, is_known)
            add_records(selected, xs, ys, is_known)
            if (index + 1) % 100 == 0:
                print(
                    f"records known={is_known} index={index+1} "
                    f"background={len(background)} selected={len(selected)}",
                    flush=True,
                )
    kh, kw, _ = KERNELS[args.kernel]
    bg_dimension = (10 if args.background_features == "full" else 2) * kh * kw
    bg_x, bg_c = to_arrays(background, bg_dimension)
    sel_x, sel_c = to_arrays(selected, 3 * kh * kw)

    reports = []
    output_dir = HERE / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant, power in enumerate(float(x) for x in args.count_powers.split(",")):
        bg_w, bg_b, bg_report = fit_classifier(
            bg_x, bg_c, power, args.known_weight, args.l2, seed_values[0] + variant
        )
        sel_w, sel_b, sel_report = fit_classifier(
            sel_x, sel_c, power, args.known_weight, args.l2, seed_values[-1] + variant
        )
        model = make_model(
            args.kernel,
            args.background_features,
            bg_w,
            bg_b,
            sel_w,
            sel_b,
            args.scale,
        )
        filename = f"task192_{args.prefix}_{args.kernel}_p{str(power).replace('.', 'p')}.onnx"
        output = output_dir / filename
        onnx.save(model, output)
        known_result = evaluate(model, known)
        train_result = evaluate(model, fresh)
        row = {
            "path": str(output.relative_to(ROOT)),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "kernel": args.kernel,
            "background_features": args.background_features,
            "count_power": power,
            "known_weight": args.known_weight,
            "l2": args.l2,
            "scale": args.scale,
            "background_fit": bg_report,
            "selected_fit": sel_report,
            "known": known_result,
            "fresh_training": train_result,
        }
        reports.append(row)
        print(json.dumps(row), flush=True)
    report_path = HERE / f"train_{args.prefix}_{args.kernel}.json"
    report_path.write_text(
        json.dumps(
            {
                "task": 192,
                "kernel": args.kernel,
                "background_features": args.background_features,
                "fresh_training_examples": len(fresh),
                "training_seeds": seed_values,
                "variants": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
