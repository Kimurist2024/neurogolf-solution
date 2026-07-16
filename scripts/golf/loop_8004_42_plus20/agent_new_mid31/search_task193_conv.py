#!/usr/bin/env python3
"""Search honest smaller single-Conv task193 models from generator states."""

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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def examples(count: int, seed: int) -> list[dict[str, np.ndarray]]:
    rows: list[dict[str, np.ndarray]] = []
    known = scoring.load_examples(193)
    for split in ("train", "test", "arc-gen"):
        for example in known[split]:
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                rows.append(converted)
    mapping = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    generator = importlib.import_module(f"task_{mapping['193']}")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    while len(rows) < sum(len(known[s]) for s in ("train", "test", "arc-gen")) + count:
        converted = scoring.convert_to_numpy(generator.generate())
        if converted is not None:
            rows.append(converted)
    return rows


def constraints(
    rows: list[dict[str, np.ndarray]], kh: int, kw: int, pads: tuple[int, int, int, int]
) -> tuple[list[dict[bytes, int]], list[dict[str, int]]]:
    top, left, bottom, right = pads
    by_channel: list[dict[bytes, int]] = [dict() for _ in range(10)]
    conflicts: list[dict[str, int]] = []
    for case_index, row in enumerate(rows):
        x = row["input"][0]
        y = row["output"][0] > 0
        xp = np.pad(x, ((0, 0), (top, bottom), (left, right)))
        for channel in range(10):
            windows = np.lib.stride_tricks.sliding_window_view(xp[channel], (kh, kw))
            assert windows.shape == (30, 30, kh, kw), windows.shape
            flat = windows.reshape(900, kh * kw).astype(np.uint8)
            labels = y[channel].reshape(900).astype(np.uint8)
            table = by_channel[channel]
            for pattern, label in zip(flat, labels, strict=True):
                key = pattern.tobytes()
                old = table.get(key)
                if old is None:
                    table[key] = int(label)
                elif old != int(label):
                    conflicts.append(
                        {
                            "case": case_index,
                            "channel": channel,
                            "old": old,
                            "new": int(label),
                        }
                    )
                    return by_channel, conflicts
    return by_channel, conflicts


def solve(table: dict[bytes, int], n: int, use_bias: bool = True) -> np.ndarray | None:
    patterns = np.asarray([np.frombuffer(key, dtype=np.uint8) for key in table], dtype=np.float64)
    labels = np.asarray(list(table.values()), dtype=np.int8)
    design = (
        np.concatenate([patterns, np.ones((len(patterns), 1))], axis=1)
        if use_bias
        else patterns
    )
    # Demand a unit signed margin.  This is stronger than the grader's raw>0 gate.
    signs = np.where(labels > 0, 1.0, -1.0)
    a_ub = -(signs[:, None] * design)
    b_ub = -np.ones(len(signs), dtype=np.float64)
    result = linprog(
        np.zeros(design.shape[1]),
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * design.shape[1],
        method="highs",
    )
    return None if not result.success else result.x


def build(weights: np.ndarray, bias: np.ndarray | None, pads: tuple[int, int, int, int], out: Path) -> None:
    kh, kw = weights.shape[-2:]
    node = helper.make_node(
        "Conv",
        ["input", "W"] + (["B"] if bias is not None else []),
        ["output"],
        group=10,
        kernel_shape=[kh, kw],
        pads=list(pads),
    )
    graph = helper.make_graph(
        [node],
        "task193_honest_single_conv",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [numpy_helper.from_array(weights.astype(np.float32), "W")]
        + ([] if bias is None else [numpy_helper.from_array(bias.astype(np.float32), "B")]),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    out.write_bytes(model.SerializeToString())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=19_300_031)
    args = parser.parse_args()
    rows = examples(args.fresh, args.seed)
    report: dict[str, object] = {"fresh_training": args.fresh, "seed": args.seed, "attempts": []}
    candidates = HERE / "candidates"
    candidates.mkdir(exist_ok=True)
    for kh in range(1, 5):
        for kw in range(1, 5):
            if kh * kw > 16:
                continue
            for top in range(kh):
                bottom = kh - 1 - top
                for left in range(kw):
                    right = kw - 1 - left
                    pads = (top, left, bottom, right)
                    tables, conflicts = constraints(rows, kh, kw, pads)
                    attempt: dict[str, object] = {
                        "kernel": [kh, kw],
                        "pads": list(pads),
                        "unique_patterns": [len(table) for table in tables],
                        "conflicts": conflicts[:1],
                    }
                    if conflicts:
                        attempt["status"] = "pattern_conflict"
                        report["attempts"].append(attempt)
                        continue
                    use_bias = kh * kw < 16
                    solutions = [solve(table, kh * kw, use_bias=use_bias) for table in tables]
                    if any(solution is None for solution in solutions):
                        attempt["status"] = "not_linearly_separable"
                        report["attempts"].append(attempt)
                        continue
                    matrix = np.stack(solutions)
                    weights = (
                        matrix[:, :-1].reshape(10, 1, kh, kw)
                        if use_bias
                        else matrix.reshape(10, 1, kh, kw)
                    )
                    bias = matrix[:, -1] if use_bias else None
                    suffix = "bias" if use_bias else "nobias"
                    out = candidates / f"task193_conv_{kh}x{kw}_p{top}{left}{bottom}{right}_{suffix}.onnx"
                    build(weights, bias, pads, out)
                    params = int(weights.size + (0 if bias is None else bias.size))
                    attempt.update(status="built", path=str(out.relative_to(ROOT)), params=params, use_bias=use_bias)
                    report["attempts"].append(attempt)
                    print("BUILT", out.name, "params", params, flush=True)
    (HERE / "task193_conv_search.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
