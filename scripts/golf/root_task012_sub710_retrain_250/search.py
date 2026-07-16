#!/usr/bin/env python3
"""Reproducible task012 7x8/7x9 depthwise-Conv census and POLICY90 MILP.

This deliberately does not repeat the historical 2.45M exact generator-domain
alignment search.  It adds two missing analyses:

* exact feasibility on the complete current 265-case corpus for every dense
  7x8/7x9 orientation and asymmetric padding; and
* a case-level MILP that maximizes complete-grid correctness, rather than
  pixel accuracy, for selected high-value padding layouts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
KNOWN_PATH = ROOT / "inputs/neurogolf-2026/task012.json"
SOURCE = ROOT / (
    "scripts/golf/loop_8004_42_plus20/agent_high47/candidates/"
    "task012_history_r01_static500_a3640a1525.onnx"
)
AUTHORITY = ROOT / "submission.zip"
sys.path.insert(0, str(TASK_DIR))
GEN = importlib.import_module("task_0962bcdd")


def known_cases() -> list[dict]:
    data = json.loads(KNOWN_PATH.read_text())
    return [item for split in ("train", "test", "arc-gen") for item in data[split]]


def domain_cases() -> list[dict]:
    return [
        GEN.generate(colors=[1, 2], cols=[col0, col1], gravity=gravity)
        for col0 in range(3, 10)
        for col1 in range(3, 10)
        for gravity in range(4)
    ]


def encode(grid: list[list[int]], channel: int) -> np.ndarray:
    values = np.asarray(grid, dtype=np.int8)
    result = np.zeros((30, 30), dtype=np.uint8)
    result[: values.shape[0], : values.shape[1]] = values == channel
    return result


def role_channels(example: dict) -> tuple[int, int]:
    values = np.asarray(example["input"], dtype=np.int8)
    colors, counts = np.unique(values[values > 0], return_counts=True)
    center = int(colors[int(np.argmin(counts))])
    arm = int(colors[int(np.argmax(counts))])
    if sorted(counts.tolist()) != [2, 8]:
        raise ValueError(f"unexpected task012 color counts {counts.tolist()}")
    return center, arm


def effective_size(kh: int, kw: int, dh: int, dw: int) -> tuple[int, int]:
    return (kh - 1) * dh + 1, (kw - 1) * dw + 1


def patches(
    mask: np.ndarray,
    kh: int,
    kw: int,
    pt: int,
    pl: int,
    dh: int = 1,
    dw: int = 1,
) -> np.ndarray:
    effh, effw = effective_size(kh, kw, dh, dw)
    padded = np.pad(mask, ((pt, effh - 1 - pt), (pl, effw - 1 - pl)))
    view = np.lib.stride_tricks.sliding_window_view(padded, (effh, effw))
    return np.ascontiguousarray(
        view[:, :, ::dh, ::dw].reshape(900, kh * kw), dtype=np.uint8
    )


def unique_channel_constraints(
    cases: list[dict],
    channel: int,
    kh: int,
    kw: int,
    pt: int,
    pl: int,
    dh: int = 1,
    dw: int = 1,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    seen: dict[bytes, int] = {}
    for example in cases:
        x = patches(encode(example["input"], channel), kh, kw, pt, pl, dh, dw)
        y = encode(example["output"], channel).reshape(-1)
        for patch, label in zip(x, y, strict=True):
            key = np.packbits(patch).tobytes()
            current = int(label)
            previous = seen.get(key)
            if previous is not None and previous != current:
                return np.empty((0, kh * kw)), np.empty(0), "contradictory_patch"
            seen[key] = current
    keys = list(seen)
    x = np.unpackbits(
        np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(len(keys), -1), axis=1
    )[:, : kh * kw].astype(np.float64)
    y = np.asarray([seen[key] for key in keys], dtype=np.float64)
    return x, y, None


def hard_linear_feasible(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray | None, dict]:
    sign = y * 2.0 - 1.0
    augmented = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    result = linprog(
        np.zeros(augmented.shape[1]),
        A_ub=-(sign[:, None] * augmented),
        b_ub=-np.ones(len(x)),
        bounds=[(None, None)] * augmented.shape[1],
        method="highs",
    )
    info = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "constraint_count": int(len(x)),
    }
    return (result.x if result.success else None), info


def dense_census() -> dict:
    cases = known_cases()
    rows = []
    for kh, kw in ((7, 8), (7, 9), (8, 7), (9, 7)):
        for pt in range(kh):
            for pl in range(kw):
                channel_rows = []
                for channel in range(10):
                    x, y, error = unique_channel_constraints(
                        cases, channel, kh, kw, pt, pl
                    )
                    if error:
                        channel_rows.append(
                            {"channel": channel, "success": False, "reason": error}
                        )
                        continue
                    _, info = hard_linear_feasible(x, y)
                    channel_rows.append({"channel": channel, **info})
                rows.append(
                    {
                        "kh": kh,
                        "kw": kw,
                        "pad_top": pt,
                        "pad_left": pl,
                        "feasible_channels": sum(row["success"] for row in channel_rows),
                        "all_channels_feasible": all(row["success"] for row in channel_rows),
                        "channels": channel_rows,
                    }
                )
    dilation_counts = {}
    for kh, kw in ((7, 8), (7, 9), (8, 7), (9, 7)):
        count = 0
        combinations = 0
        for dh in range(1, 30):
            for dw in range(1, 30):
                effh, effw = effective_size(kh, kw, dh, dw)
                if effh <= 30 and effw <= 30:
                    combinations += 1
                    count += effh * effw
        dilation_counts[f"{kh}x{kw}"] = {
            "dilation_pairs": combinations,
            "nonnegative_asymmetric_pad_layouts": count,
        }
    return {
        "task": 12,
        "known_total": len(cases),
        "dense_layout_count": len(rows),
        "dense_all_channel_feasible": sum(row["all_channels_feasible"] for row in rows),
        "best_feasible_channel_count": max(row["feasible_channels"] for row in rows),
        "rows": rows,
        "dilated_geometry_census": dilation_counts,
        "historical_exact_search_not_repeated": {
            "script": "scripts/golf/scratch_codex/task012/reference_and_dilated_search.py",
            "nonnegative_dilated_attempts": 2447544,
            "result": "NO FEASIBLE dilated group10 under70",
            "shifted_dense_attempts": 218962,
            "shifted_dilated_high_area_attempts": 702546,
        },
    }


def role_state_constraints(
    example: dict,
    kh: int,
    kw: int,
    pt: int,
    pl: int,
) -> tuple[np.ndarray, np.ndarray]:
    center, arm = role_channels(example)
    seen: dict[bytes, int] = {}
    for channel in (center, arm):
        x = patches(encode(example["input"], channel), kh, kw, pt, pl)
        y = encode(example["output"], channel).reshape(-1)
        for patch, label in zip(x, y, strict=True):
            key = np.packbits(patch).tobytes()
            current = int(label)
            previous = seen.get(key)
            if previous is not None and previous != current:
                raise RuntimeError("one state contains a contradictory role patch")
            seen[key] = current
    keys = list(seen)
    x = np.unpackbits(
        np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(len(keys), -1), axis=1
    )[:, : kh * kw].astype(np.float64)
    return x, np.asarray([seen[key] for key in keys], dtype=np.float64)


def maximize_exact_cases(
    cases: list[dict],
    kh: int,
    kw: int,
    pt: int,
    pl: int,
    *,
    time_limit: float,
) -> tuple[dict, np.ndarray | None]:
    states = [role_state_constraints(case, kh, kw, pt, pl) for case in cases]
    dimension = kh * kw
    state_count = len(states)
    row_count = sum(len(y) for _, y in states)
    coefficient_bound = 100.0
    big_m = (dimension + 1) * coefficient_bound + 1.0
    matrix = lil_matrix((row_count, dimension + 1 + state_count), dtype=np.float64)
    lower = np.full(row_count, 1.0 - big_m, dtype=np.float64)
    cursor = 0
    for state_index, (x, y) in enumerate(states):
        sign = y * 2.0 - 1.0
        count = len(y)
        matrix[cursor : cursor + count, :dimension] = sign[:, None] * x
        matrix[cursor : cursor + count, dimension] = sign
        matrix[cursor : cursor + count, dimension + 1 + state_index] = -big_m
        cursor += count
    objective = np.concatenate(
        [np.zeros(dimension + 1), -np.ones(state_count)]
    )
    bounds = Bounds(
        np.concatenate(
            [np.full(dimension + 1, -coefficient_bound), np.zeros(state_count)]
        ),
        np.concatenate(
            [np.full(dimension + 1, coefficient_bound), np.ones(state_count)]
        ),
    )
    integrality = np.concatenate(
        [np.zeros(dimension + 1), np.ones(state_count)]
    )
    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(
            matrix.tocsr(), lower, np.full(row_count, np.inf)
        ),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
    )
    info = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "state_count": state_count,
        "constraint_count": row_count,
        "best_exact_cases": (
            int(round(-float(result.fun))) if result.fun is not None else None
        ),
        "mip_gap": float(result.mip_gap) if result.mip_gap is not None else None,
        "mip_dual_bound": (
            float(result.mip_dual_bound)
            if result.mip_dual_bound is not None
            else None
        ),
    }
    return info, (result.x[: dimension + 1] if result.x is not None else None)


def embed_source_background(
    kh: int, kw: int, pt: int, pl: int
) -> tuple[np.ndarray, float]:
    model = onnx.load(SOURCE)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    row_start = pt - 3
    col_start = pl - 3
    if not (0 <= row_start <= kh - 7 and 0 <= col_start <= kw - 7):
        raise ValueError("layout cannot embed the exact 7x7 source background")
    weight = np.zeros((kh, kw), dtype=np.float32)
    weight[row_start : row_start + 7, col_start : col_start + 7] = arrays["w"][0, 0]
    return weight, float(arrays["b"][0])


def make_model(
    kh: int,
    kw: int,
    pt: int,
    pl: int,
    foreground: np.ndarray,
    output: Path,
) -> None:
    background_w, background_b = embed_source_background(kh, kw, pt, pl)
    weights = np.empty((10, 1, kh, kw), dtype=np.float32)
    weights[0, 0] = background_w
    weights[1:, 0] = foreground[:-1].reshape(kh, kw).astype(np.float32)
    bias = np.asarray(
        [background_b] + [float(foreground[-1])] * 9, dtype=np.float32
    )
    node = helper.make_node(
        "Conv",
        ["input", "w", "b"],
        ["output"],
        group=10,
        kernel_shape=[kh, kw],
        pads=[pt, pl, kh - 1 - pt, kw - 1 - pl],
    )
    graph = helper.make_graph(
        [node],
        "task012_policy90_depthwise_conv",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=[
            numpy_helper.from_array(weights, "w"),
            numpy_helper.from_array(bias, "b"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
        producer_name="root_task012_sub710_retrain_250",
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    census_parser = subparsers.add_parser("census")
    census_parser.add_argument("--output", type=Path, default=HERE / "dense_census.json")
    milp_parser = subparsers.add_parser("milp")
    milp_parser.add_argument("--dataset", choices=("known", "domain"), required=True)
    milp_parser.add_argument("--kh", type=int, required=True)
    milp_parser.add_argument("--kw", type=int, required=True)
    milp_parser.add_argument("--pt", type=int, required=True)
    milp_parser.add_argument("--pl", type=int, required=True)
    milp_parser.add_argument("--time-limit", type=float, default=120.0)
    milp_parser.add_argument("--output", type=Path, required=True)
    milp_parser.add_argument("--model", type=Path)
    args = parser.parse_args()

    if args.command == "census":
        result = dense_census()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "dense_layout_count": result["dense_layout_count"],
                    "dense_all_channel_feasible": result["dense_all_channel_feasible"],
                    "best_feasible_channel_count": result["best_feasible_channel_count"],
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0

    cases = known_cases() if args.dataset == "known" else domain_cases()
    info, solution = maximize_exact_cases(
        cases,
        args.kh,
        args.kw,
        args.pt,
        args.pl,
        time_limit=args.time_limit,
    )
    result = {
        "task": 12,
        "dataset": args.dataset,
        "kh": args.kh,
        "kw": args.kw,
        "pad_top": args.pt,
        "pad_left": args.pl,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "solver": info,
        "foreground": solution.tolist() if solution is not None else None,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.model and solution is not None:
        make_model(args.kh, args.kw, args.pt, args.pl, solution, args.model)
        result["model"] = str(args.model.resolve())
        result["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
