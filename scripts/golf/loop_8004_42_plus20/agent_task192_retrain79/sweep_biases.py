#!/usr/bin/env python3
"""Tune the two semantic Conv biases without changing task192 structure.

For each generator case, exactness is an interval constraint on the channel-0
bias and another interval constraint on the dynamically selected-color bias.
This script searches those two scalar offsets while requiring all 265 known
cases to remain exact, then emits only the best fixed-SHA variants.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))


def tensors(example: dict[str, object]) -> tuple[np.ndarray, np.ndarray, int]:
    grid = np.asarray(example["input"], dtype=np.int8)
    output = np.asarray(example["output"], dtype=np.int8)
    height, width = grid.shape
    input_tensor = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for color in range(10):
        input_tensor[0, color, :height, :width] = grid == color
    expected = np.full((30, 30), -1, dtype=np.int8)
    expected[:height, :width] = output
    flat = grid.reshape(-1).tolist()
    selected = max(range(1, 10), key=flat.count)
    return input_tensor, expected, selected


def intervals(
    inference: ort.InferenceSession, examples: list[dict[str, object]]
) -> np.ndarray:
    rows = []
    for example in examples:
        input_tensor, expected, selected = tensors(example)
        raw = inference.run(None, {"input": input_tensor})[0][0]
        expected0 = expected == 0
        expected_selected = expected == selected
        # A positive cell needs delta > -raw; a false cell needs delta <= -raw.
        low0 = float(-raw[0][expected0].min()) if expected0.any() else -np.inf
        high0 = float(-raw[0][~expected0].max()) if (~expected0).any() else np.inf
        lows = (
            float(-raw[selected][expected_selected].min())
            if expected_selected.any()
            else -np.inf
        )
        highs = (
            float(-raw[selected][~expected_selected].max())
            if (~expected_selected).any()
            else np.inf
        )
        rows.append((low0, high0, lows, highs))
    return np.asarray(rows, dtype=np.float64)


def covered(values: np.ndarray, points: np.ndarray, low_col: int, high_col: int) -> np.ndarray:
    return (points[None, :] > values[:, low_col, None]) & (
        points[None, :] <= values[:, high_col, None]
    )


def initializer(model: onnx.ModelProto, name: str) -> tuple[int, np.ndarray]:
    for index, item in enumerate(model.graph.initializer):
        if item.name == name:
            return index, numpy_helper.to_array(item).copy()
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--seeds", default="79192911,79192912")
    parser.add_argument("--fresh", type=int, default=500)
    parser.add_argument("--step", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--prefix", default="bias")
    args = parser.parse_args()

    model = onnx.load(args.model)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    inference = ort.InferenceSession(model.SerializeToString(), options)
    known_data = json.loads((ROOT / "inputs/neurogolf-2026/task192.json").read_text())
    known_examples = known_data["train"] + known_data["test"] + known_data["arc-gen"]
    known = intervals(inference, known_examples)
    allowed = {
        "background": [float(known[:, 0].max()), float(known[:, 1].min())],
        "selected": [float(known[:, 2].max()), float(known[:, 3].min())],
    }
    if not (
        allowed["background"][0] < allowed["background"][1]
        and allowed["selected"][0] < allowed["selected"][1]
    ):
        raise RuntimeError(f"known set has no common exact bias interval: {allowed}")

    generator = importlib.import_module("task_7e0986d6")
    seed_values = [int(value) for value in args.seeds.split(",")]
    fresh_by_seed = []
    for seed in seed_values:
        random.seed(seed)
        examples = [generator.generate() for _ in range(args.fresh)]
        fresh_by_seed.append(intervals(inference, examples))
        print(f"generated/audited seed={seed} count={len(examples)}", flush=True)
    fresh = np.concatenate(fresh_by_seed)
    seed_id = np.repeat(np.arange(len(seed_values)), args.fresh)

    def point_grid(bounds: list[float]) -> np.ndarray:
        lower = np.floor(bounds[0] / args.step) * args.step + args.step
        upper = np.floor(bounds[1] / args.step) * args.step
        values = np.arange(lower, upper + args.step / 2, args.step)
        points = np.unique(np.r_[values, 0.0])
        return points[(points > bounds[0]) & (points <= bounds[1])]

    background_points = point_grid(allowed["background"])
    selected_points = point_grid(allowed["selected"])
    background_cover = covered(fresh, background_points, 0, 1)
    selected_cover = covered(fresh, selected_points, 2, 3)
    results = []
    for bg_index, bg_delta in enumerate(background_points):
        active = background_cover[:, bg_index]
        joint = selected_cover & active[:, None]
        per_seed = np.stack(
            [joint[seed_id == index].sum(axis=0) for index in range(len(seed_values))]
        )
        minimum = per_seed.min(axis=0)
        total = per_seed.sum(axis=0)
        selected_order = np.lexsort((-total, -minimum))
        for selected_index in selected_order[: min(args.top, len(selected_points))]:
            results.append(
                {
                    "background_delta": float(bg_delta),
                    "selected_delta": float(selected_points[selected_index]),
                    "right_by_seed": [int(x) for x in per_seed[:, selected_index]],
                    "minimum_accuracy": float(minimum[selected_index] / args.fresh),
                    "total_right": int(total[selected_index]),
                }
            )
    results.sort(key=lambda row: (row["minimum_accuracy"], row["total_right"]), reverse=True)
    unique = []
    seen = set()
    for row in results:
        key = (row["background_delta"], row["selected_delta"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
        if len(unique) >= args.top:
            break

    bias_base_index, bias_base = initializer(model, "bias_base")
    bias_update_index, bias_update = initializer(model, "bias_update")
    candidates = []
    for rank, row in enumerate(unique, 1):
        candidate = onnx.ModelProto()
        candidate.CopyFrom(model)
        changed_base = bias_base.copy()
        changed_update = bias_update.copy()
        changed_base[0] += np.float32(row["background_delta"])
        changed_update[0] += np.float32(row["selected_delta"])
        candidate.graph.initializer[bias_base_index].CopyFrom(
            numpy_helper.from_array(changed_base, "bias_base")
        )
        candidate.graph.initializer[bias_update_index].CopyFrom(
            numpy_helper.from_array(changed_update, "bias_update")
        )
        onnx.checker.check_model(candidate, full_check=True)
        onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
        output = HERE / "candidates" / f"task192_{args.prefix}_r{rank:02d}.onnx"
        onnx.save(candidate, output)
        item = dict(row)
        item.update(
            path=str(output.relative_to(ROOT)),
            sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        )
        candidates.append(item)
        print(json.dumps(item), flush=True)
    report = {
        "task": 192,
        "source": str(args.model),
        "fresh_per_seed": args.fresh,
        "seeds": seed_values,
        "known_exact_delta_intervals": allowed,
        "step": args.step,
        "candidates": candidates,
    }
    (HERE / f"sweep_{args.prefix}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
