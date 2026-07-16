#!/usr/bin/env python3
"""Search exact bias-free depthwise Conv classifiers for task012.

Unlike the earlier hard-margin sweep, an all-zero negative patch is allowed to
score exactly zero because the competition decoder uses ``raw > 0``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "lane_c4" / "search_task012_lp.py"
SPEC = importlib.util.spec_from_file_location("task012_old_search", SOURCE)
assert SPEC is not None and SPEC.loader is not None
OLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OLD)


def homogeneous_solve(
    x: np.ndarray, y: np.ndarray, strict_nonzero_negatives: bool
) -> tuple[np.ndarray | None, dict[str, object]]:
    positive = y > 0
    nonzero = np.any(x != 0, axis=1)
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    if np.any(positive):
        rows.extend(-x[positive])
        bounds.extend([-1.0] * int(positive.sum()))
    negative = ~positive
    if strict_nonzero_negatives:
        strict = negative & nonzero
        rows.extend(x[strict])
        bounds.extend([-1.0] * int(strict.sum()))
    else:
        rows.extend(x[negative])
        bounds.extend([0.0] * int(negative.sum()))
    aub = np.asarray(rows, dtype=np.float64)
    bub = np.asarray(bounds, dtype=np.float64)
    result = linprog(
        np.zeros(x.shape[1]), A_ub=aub, b_ub=bub,
        bounds=[(None, None)] * x.shape[1], method="highs",
    )
    info: dict[str, object] = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "constraint_count": int(len(aub)),
        "strict_nonzero_negatives": strict_nonzero_negatives,
    }
    if not result.success:
        return None, info
    w = result.x
    scores = x @ w
    info.update(
        positive_min=float(scores[positive].min()),
        negative_max=float(scores[negative].max()),
        nonzero_negative_max=float(scores[negative & nonzero].max()),
        maximum_abs_coefficient=float(np.abs(w).max()),
    )
    return w, info


def make_model(
    kh: int, kw: int, pt: int, pl: int, background: np.ndarray, foreground: np.ndarray
) -> onnx.ModelProto:
    weights = np.empty((10, 1, kh, kw), dtype=np.float32)
    weights[0, 0] = background.reshape(kh, kw)
    weights[1:, 0] = foreground.reshape(kh, kw)
    node = helper.make_node(
        "Conv", ["input", "w"], ["output"], group=10,
        kernel_shape=[kh, kw], pads=[pt, pl, kh - 1 - pt, kw - 1 - pl],
    )
    graph = helper.make_graph(
        [node], "task012_homogeneous_exact",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=[numpy_helper.from_array(weights, "w")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10,
        producer_name="task012-homogeneous-search",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int, default=70)
    parser.add_argument("--min-h", type=int, default=1)
    parser.add_argument("--max-h", type=int, default=10)
    parser.add_argument("--min-w", type=int, default=1)
    parser.add_argument("--max-w", type=int, default=10)
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="scan only the kh<7 or kw<7 gap in the earlier biased sweep",
    )
    parser.add_argument(
        "--long-side-only",
        action="store_true",
        help="scan only shapes with a side above ten",
    )
    parser.add_argument("--weak-negatives", action="store_true")
    args = parser.parse_args()
    examples = OLD.exhaustive_examples()
    results: list[dict[str, object]] = []
    winners: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    strict = not args.weak_negatives
    for area in range(args.min_area, args.max_area + 1):
        for kh in range(args.min_h, args.max_h + 1):
            for kw in range(args.min_w, args.max_w + 1):
                if kh * kw != area:
                    continue
                if args.missing_only and kh >= 7 and kw >= 7:
                    continue
                if args.long_side_only and max(kh, kw) <= 10:
                    continue
                for pt in range(kh):
                    for pl in range(kw):
                        row: dict[str, object] = {
                            "kh": kh, "kw": kw, "pad_top": pt, "pad_left": pl
                        }
                        sols: list[np.ndarray] = []
                        for channel, label in ((0, "background"), (1, "foreground")):
                            x, y, error = OLD.unique_constraints(
                                examples, channel, kh, kw, pt, pl
                            )
                            if error:
                                row[label] = {"success": False, "reason": error}
                                break
                            sol, info = homogeneous_solve(x, y, strict)
                            row[label] = info
                            if sol is None:
                                break
                            sols.append(sol)
                        row["success"] = len(sols) == 2
                        results.append(row)
                        if len(sols) == 2:
                            winners.append((kh, kw, pt, pl, sols[0], sols[1]))
                            print("WIN", kh, kw, pt, pl, "params", 10 * area, flush=True)
        if winners:
            break
        if area % 5 == 0:
            print("searched area", area, flush=True)

    output = {
        "domain_examples": len(examples),
        "strict_nonzero_negatives": strict,
        "missing_only": args.missing_only,
        "long_side_only": args.long_side_only,
        "searched_through_area": area,
        "results": results,
    }
    suffix = "strict" if strict else "weak"
    if args.long_side_only:
        suffix += "_long_side"
    (HERE / f"homogeneous_search_{suffix}.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    if not winners:
        print("no feasible homogeneous window")
        return 1
    for kh, kw, pt, pl, bg, fg in winners:
        model = make_model(kh, kw, pt, pl, bg, fg)
        path = HERE / "candidates" / f"task012_h{kh}w{kw}_pt{pt}pl{pl}_nobias.onnx"
        onnx.save(model, path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
