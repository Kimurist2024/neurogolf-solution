#!/usr/bin/env python3
"""Refute full-support sign equivalence for task344 compact-G cost132.

The witness is not an arbitrary four-color grid.  It is represented by an
explicit latent outcome of task_d90796e8.generate(): a Bernoulli gray subset,
a Bernoulli subset of padded red centers satisfying the generator's greedy
spacing rule, and one legal decoration outcome for every accepted center.
Every listed random choice has strictly positive probability.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REBASE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_task344_rebase_171"
AUTHORITY = REBASE / "baseline/task344.onnx"
CANDIDATE = REBASE / "candidates/task344_compact_g_cost132.onnx"
GENERATOR_PATH = ROOT / "inputs/arc-gen-repo/tasks/task_d90796e8.py"
EXPECTED_AUTHORITY_SHA256 = "05bedf3ca834aadfc973c00fc91cafdb4d0ae1aaab374115d924e2e33fb1bf6c"
EXPECTED_CANDIDATE_SHA256 = "c5272a42bee419008a15d14bea734a6fb15956a863ad8e702deac0f02fcea5f6"
HEIGHT = WIDTH = 10
TARGET = (5, 5)
TARGET_CHANNEL = 8

# Center coordinates use the generator's padded coordinate system.  The last
# center is outside the visible grid and draws its green pixel inward.
CENTERS_AND_DECORATIONS = [
    ((1, 3), "down"),
    ((1, 8), "down"),
    ((2, 1), None),
    ((2, 6), "down"),
    ((3, 4), "down"),
    ((3, 9), "left"),
    ((4, 2), "right"),
    ((4, 7), "down"),
    ((5, 0), "right"),
    ((5, 5), None),
    ((6, 3), None),
    ((6, 8), "up"),
    ((7, 1), "down"),
    ((7, 6), "right"),
    ((8, 4), "up"),
    ((8, 9), None),
    ((9, 7), "up"),
    ((10, 3), "up"),
]
GRAY_SUBSET = [
    (1, 4),
    (1, 5),
    (2, 2),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 3),
    (5, 4),
    (5, 6),
    (6, 5),
    (6, 6),
    (6, 7),
    (7, 5),
    (8, 5),
    (8, 6),
    (8, 8),
]
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
EXPECTED_GRID = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 5, 5, 0, 0, 3, 0],
        [0, 3, 5, 2, 0, 0, 3, 0, 2, 0],
        [0, 0, 0, 0, 3, 0, 2, 5, 2, 3],
        [0, 0, 3, 2, 2, 5, 5, 3, 0, 0],
        [3, 2, 0, 5, 5, 3, 5, 2, 2, 0],
        [0, 0, 0, 3, 0, 5, 5, 5, 3, 0],
        [0, 3, 0, 0, 2, 5, 3, 2, 0, 0],
        [0, 2, 0, 0, 3, 5, 5, 2, 5, 3],
        [0, 0, 0, 2, 0, 0, 0, 3, 0, 0],
    ],
    dtype=np.int64,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(grid: np.ndarray, row: int, col: int, color: int) -> None:
    if 0 <= row < HEIGHT and 0 <= col < WIDTH:
        grid[row, col] = color


def reconstruct_witness() -> tuple[np.ndarray, dict]:
    centers = [center for center, _ in CENTERS_AND_DECORATIONS]
    assert centers == sorted(centers), "random_pixels visits padded centers row-major"
    assert len(set(centers)) == len(centers)
    assert all(-1 <= row <= HEIGHT and -1 <= col <= WIDTH for row, col in centers)
    distances = [
        abs(row_a - row_b) + abs(col_a - col_b)
        for index, (row_a, col_a) in enumerate(centers)
        for row_b, col_b in centers[index + 1 :]
    ]
    minimum_distance = min(distances)
    assert minimum_distance > 2

    # An exact initial gray subset is a possible random_pixels outcome.
    grid = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    for row, col in GRAY_SUBSET:
        assert 0 <= row < HEIGHT and 0 <= col < WIDTH
        grid[row, col] = 5

    # With no other padded candidates selected, the greedy overlap filter
    # accepts every listed center because all pairwise distances exceed two.
    for (row, col), decoration in CENTERS_AND_DECORATIONS:
        draw(grid, row, col, 3)
        if decoration is not None:
            dr, dc = DIRECTIONS[decoration]
            draw(grid, row + dr, col + dc, 2)
    assert np.array_equal(grid, EXPECTED_GRID)

    gray_count = len(GRAY_SUBSET)
    padded_count = (HEIGHT + 2) * (WIDTH + 2)
    center_count = len(centers)
    decorated_count = sum(decoration is not None for _, decoration in CENTERS_AND_DECORATIONS)
    undecorated_count = center_count - decorated_count
    log_probability = (
        2 * math.log(1 / 8)  # independently chosen dimensions 3..10
        + gray_count * math.log(0.04)
        + (HEIGHT * WIDTH - gray_count) * math.log(0.96)
        + center_count * math.log(0.08)
        + (padded_count - center_count) * math.log(0.92)
        + decorated_count * math.log(3 / 16)
        + undecorated_count * math.log(1 / 4)
    )
    support = {
        "dimensions": [HEIGHT, WIDTH],
        "gray_subset_count": gray_count,
        "padded_candidate_count": padded_count,
        "selected_center_count": center_count,
        "decorated_center_count": decorated_count,
        "undecorated_center_count": undecorated_count,
        "minimum_pairwise_center_manhattan_distance": minimum_distance,
        "greedy_spacing_accepts_all": True,
        "exact_latent_outcome_log_probability": log_probability,
        "exact_latent_outcome_probability_is_strictly_positive": math.isfinite(log_probability),
        "reason": (
            "Every finite Bernoulli subset has positive probability; each no-green outcome has "
            "probability 1/4 and each specified cardinal green direction has probability 3/16."
        ),
    }
    return grid, support


def arrays(path: Path) -> dict[str, np.ndarray]:
    return {
        item.name: numpy_helper.to_array(item).astype(np.float64)
        for item in onnx.load(path).graph.initializer
    }


def direct_target_logit(grid: np.ndarray) -> dict:
    authority = arrays(AUTHORITY)
    candidate = arrays(CANDIDATE)
    authority_tensor = np.einsum(
        "sl,ld,kc,sx,xk,ky,yo->dco",
        authority["H"],
        authority["V"],
        authority["V"],
        authority["S"],
        authority["H"],
        authority["M"],
        authority["V"],
        optimize=True,
    )
    candidate_tensor = np.einsum(
        "ld,kc,lk,ky,yo->dco",
        candidate["V"],
        candidate["V"],
        candidate["G"],
        candidate["M"],
        candidate["V"],
        optimize=True,
    )
    kernel = np.power(authority["B"].T @ authority["B"], 16)
    row, col = TARGET
    weights = np.outer(kernel[:HEIGHT, row], kernel[:WIDTH, col])
    center_color = int(grid[row, col])
    authority_logit = float(np.sum(weights * authority_tensor[grid, center_color, TARGET_CHANNEL]))
    candidate_logit = float(np.sum(weights * candidate_tensor[grid, center_color, TARGET_CHANNEL]))
    return {
        "evaluator": "float64 contraction of serialized float32 initializers",
        "center_color": center_color,
        "authority": authority_logit,
        "candidate": candidate_logit,
        "authority_positive": authority_logit > 0,
        "candidate_positive": candidate_logit > 0,
        "sign_mismatch": (authority_logit > 0) != (candidate_logit > 0),
    }


def strict_model_check(path: Path) -> dict:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return {"checker_full": True, "strict_shape_inference_data_prop": True}


def main() -> None:
    assert sha256(AUTHORITY) == EXPECTED_AUTHORITY_SHA256
    assert sha256(CANDIDATE) == EXPECTED_CANDIDATE_SHA256
    grid, support = reconstruct_witness()

    sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo"))
    sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
    generator = importlib.import_module("task_d90796e8")
    rows: list[int] = []
    cols: list[int] = []
    colors: list[int] = []
    for row in range(HEIGHT):
        for col in range(WIDTH):
            if grid[row, col] != 0:
                rows.append(row)
                cols.append(col)
                colors.append(int(grid[row, col]))
    explicit = generator.generate(
        width=WIDTH,
        height=HEIGHT,
        rows=rows,
        cols=cols,
        colors=colors,
    )
    assert explicit["input"] == grid.tolist()

    sys.path.insert(0, str(ROOT))
    from scripts.golf.rank_dir import cost_of
    from scripts.lib import scoring

    bench = scoring.convert_to_numpy(explicit)
    modes = [
        ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
        ("extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
        ("enable_all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ]
    runtime: list[dict] = []
    authority_model = scoring.sanitize_model(copy.deepcopy(onnx.load(AUTHORITY)))
    candidate_model = scoring.sanitize_model(copy.deepcopy(onnx.load(CANDIDATE)))
    for name, level in modes:
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 4
        authority_session = ort.InferenceSession(authority_model.SerializeToString(), options)
        candidate_session = ort.InferenceSession(candidate_model.SerializeToString(), options)
        authority_output = authority_session.run(["output"], {"input": bench["input"]})[0]
        candidate_output = candidate_session.run(["output"], {"input": bench["input"]})[0]
        mismatch = (authority_output > 0) != (candidate_output > 0)
        indices = np.argwhere(mismatch)
        row, col = TARGET
        authority_raw = float(authority_output[0, TARGET_CHANNEL, row, col])
        candidate_raw = float(candidate_output[0, TARGET_CHANNEL, row, col])
        assert authority_raw > 0 and candidate_raw < 0
        runtime.append(
            {
                "mode": name,
                "authority_target_raw": authority_raw,
                "candidate_target_raw": candidate_raw,
                "target_sign_mismatch": True,
                "full_output_sign_mismatch_count": int(np.count_nonzero(mismatch)),
                "first_sign_mismatch_indices": indices[:20].tolist(),
                "authority_nonfinite": int(np.count_nonzero(~np.isfinite(authority_output))),
                "candidate_nonfinite": int(np.count_nonzero(~np.isfinite(candidate_output))),
            }
        )

    cost = cost_of(str(CANDIDATE))
    result = {
        "lane": "agent_task344_margin_184",
        "task": 344,
        "authority": {
            "path": str(AUTHORITY.relative_to(ROOT)),
            "sha256": sha256(AUTHORITY),
            "cost": 137,
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": sha256(CANDIDATE),
            "cost": {"memory": cost[0], "params": cost[1], "cost": cost[2]},
            "potential_score_gain": float(math.log(137 / 132)),
        },
        "generator": {
            "path": str(GENERATOR_PATH.relative_to(ROOT)),
            "sha256": sha256(GENERATOR_PATH),
            "support_witness": support,
            "explicit_generate_matches_witness_input": True,
        },
        "witness": {
            "grid": grid.tolist(),
            "target": {"row": TARGET[0], "col": TARGET[1], "channel": TARGET_CHANNEL},
            "centers_and_decorations": [
                {"center": list(center), "green_direction": decoration}
                for center, decoration in CENTERS_AND_DECORATIONS
            ],
            "gray_subset": [list(cell) for cell in GRAY_SUBSET],
            "direct_serialized_weight_evaluator": direct_target_logit(grid),
            "official_onnxruntime_modes": runtime,
        },
        "model_checks": {
            "authority": strict_model_check(AUTHORITY),
            "candidate": strict_model_check(CANDIDATE),
        },
        "proof": {
            "full_generator_support_sign_equivalence": False,
            "margin_certificate_exists_for_this_candidate": False,
            "counterexample_is_in_generator_support": True,
            "counterexample_is_not_fresh_sampling": True,
            "conclusion": (
                "The reachable witness flips authority channel-8 sign at (5,5) in all four "
                "ONNX Runtime optimization modes, so compact-G cannot be certified as a "
                "full-support sign-preserving replacement."
            ),
        },
        "winner_eligible": False,
        "probe_only": True,
        "verdict": "PROBE_ONLY",
        "protected_files_modified": False,
    }
    audit_path = HERE / "audit/margin_counterexample.json"
    audit_path.write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "result.json").write_text(
        json.dumps(
            {
                "lane": result["lane"],
                "task": 344,
                "verdict": result["verdict"],
                "winner": None,
                "probe_candidate": result["candidate"],
                "reason": result["proof"]["conclusion"],
                "root_modified": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "support": support,
                "direct": result["witness"]["direct_serialized_weight_evaluator"],
                "runtime": runtime,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
