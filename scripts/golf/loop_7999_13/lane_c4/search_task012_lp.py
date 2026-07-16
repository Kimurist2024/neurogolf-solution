#!/usr/bin/env python3
"""Find the smallest exact depthwise-Conv window for generator task0962bcdd.

The incumbent task012 graph applies one shared spatial classifier to every
non-background colour (the serialized Conv repeats that classifier for ONNX
grouped-Conv shape compliance).  This script enumerates the complete geometric
domain for colours 1/2, deduplicates binary neighbourhoods, and solves hard
linear-separation feasibility for the background and shared foreground
classifiers.  Visible examples are not used to fit the rule.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
GEN = importlib.import_module("task_0962bcdd")


def encode(grid: list[list[int]], channel: int) -> np.ndarray:
    a = np.asarray(grid, dtype=np.int8)
    out = np.zeros((30, 30), dtype=np.uint8)
    out[: a.shape[0], : a.shape[1]] = a == channel
    return out


def exhaustive_examples() -> list[dict]:
    examples: list[dict] = []
    # Both assignments make channel 1 play each non-background semantic role.
    for colors in ([1, 2], [2, 1]):
        for c0 in range(3, 10):
            for c1 in range(3, 10):
                for gravity in range(4):
                    examples.append(
                        GEN.generate(colors=colors, cols=[c0, c1], gravity=gravity)
                    )
    return examples


def unique_constraints(
    examples: list[dict], channel: int, kh: int, kw: int, pt: int, pl: int
) -> tuple[np.ndarray, np.ndarray, str | None]:
    pb, pr = kh - 1 - pt, kw - 1 - pl
    seen: dict[bytes, int] = {}
    for ex in examples:
        x = encode(ex["input"], channel)
        y = encode(ex["output"], channel)
        xp = np.pad(x, ((pt, pb), (pl, pr)))
        patches = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw)).reshape(
            900, kh * kw
        )
        labels = y.reshape(-1)
        for patch, label in zip(patches, labels):
            key = np.packbits(patch).tobytes()
            old = seen.get(key)
            cur = int(label)
            if old is not None and old != cur:
                return np.empty((0, kh * kw)), np.empty(0), "contradictory_patch"
            seen[key] = cur
    keys = list(seen)
    x = np.unpackbits(
        np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(len(keys), -1), axis=1
    )[:, : kh * kw].astype(np.float64)
    y = np.asarray([seen[k] for k in keys], dtype=np.float64)
    return x, y, None


def solve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray | None, dict]:
    # y=1 => w.x+b >= 1; y=0 => w.x+b <= -1.
    sign = np.where(y > 0, 1.0, -1.0)
    z = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    aub = -(sign[:, None] * z)
    bub = -np.ones(len(x))
    result = linprog(
        np.zeros(z.shape[1]), A_ub=aub, b_ub=bub,
        bounds=[(None, None)] * z.shape[1], method="highs",
    )
    info = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "constraint_count": int(len(x)),
    }
    if not result.success:
        return None, info
    v = result.x
    margin = sign * (z @ v)
    info["minimum_signed_margin"] = float(margin.min())
    info["maximum_abs_coefficient"] = float(np.abs(v).max())
    return v, info


def make_model(
    kh: int, kw: int, pt: int, pl: int, background: np.ndarray, foreground: np.ndarray
) -> onnx.ModelProto:
    weights = np.empty((10, 1, kh, kw), dtype=np.float32)
    weights[0, 0] = background[:-1].reshape(kh, kw)
    weights[1:, 0] = foreground[:-1].reshape(kh, kw)
    bias = np.asarray([background[-1]] + [foreground[-1]] * 9, dtype=np.float32)
    node = helper.make_node(
        "Conv", ["input", "w", "b"], ["output"], group=10,
        kernel_shape=[kh, kw], pads=[pt, pl, kh - 1 - pt, kw - 1 - pl],
    )
    graph = helper.make_graph(
        [node], "task012_generator_exact_lp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=[numpy_helper.from_array(weights, "w"), numpy_helper.from_array(bias, "b")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8,
        producer_name="task012-generator-exact-lp",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-h", type=int, default=1)
    parser.add_argument("--max-h", type=int, default=7)
    parser.add_argument("--min-w", type=int, default=1)
    parser.add_argument("--max-w", type=int, default=10)
    args = parser.parse_args()
    examples = exhaustive_examples()
    results: list[dict] = []
    winners: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    for area in range(args.min_h * args.min_w, args.max_h * args.max_w + 1):
        for kh in range(args.min_h, args.max_h + 1):
            for kw in range(args.min_w, args.max_w + 1):
                if kh * kw != area:
                    continue
                for pt in range(kh):
                    for pl in range(kw):
                        row = {"kh": kh, "kw": kw, "pad_top": pt, "pad_left": pl}
                        sols: list[np.ndarray] = []
                        for channel, label in ((0, "background"), (1, "foreground")):
                            x, y, error = unique_constraints(
                                examples, channel, kh, kw, pt, pl
                            )
                            if error:
                                row[label] = {"success": False, "reason": error}
                                break
                            sol, info = solve(x, y)
                            row[label] = info
                            if sol is None:
                                break
                            sols.append(sol)
                        row["success"] = len(sols) == 2
                        results.append(row)
                        if len(sols) == 2:
                            winners.append((kh, kw, pt, pl, sols[0], sols[1]))
                            print("WIN", kh, kw, pt, pl, "params", 10 * kh * kw + 10)
        if winners:
            break

    (HERE / "task012_lp_search.json").write_text(
        json.dumps({"domain_examples": len(examples), "results": results}, indent=2) + "\n"
    )
    if not winners:
        print("no feasible window")
        return 1
    # Emit every minimum-area alignment; runtime/gold/fresh gates choose among them.
    outdir = HERE / "task012_lp_candidates"
    outdir.mkdir(exist_ok=True)
    for kh, kw, pt, pl, bg, fg in winners:
        model = make_model(kh, kw, pt, pl, bg, fg)
        path = outdir / f"task012_h{kh}w{kw}_pt{pt}pl{pl}.onnx"
        onnx.save(model, path)
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
