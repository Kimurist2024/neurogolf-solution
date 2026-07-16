#!/usr/bin/env python3
"""Formal/support and Monte Carlo analysis for task192 fixed thresholds."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
THRESHOLDS = tuple(range(26, 37))
SEEDS = (192800661, 192930007)
SAMPLES_PER_SEED = 5_000
EXPECTED_ROOT = {
    "submission.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "all_scores.csv": "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78",
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
from lib import scoring  # noqa: E402

generator = importlib.import_module("task_7e0986d6")
common = importlib.import_module("common")
onnxruntime.set_default_logger_severity(4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def count_colors(example: dict[str, list[list[int]]]) -> tuple[int, int, int]:
    output_flat = [value for row in example["output"] for value in row]
    box_colors = set(output_flat) - {0}
    assert len(box_colors) == 1
    box_color = next(iter(box_colors))
    input_flat = [value for row in example["input"] for value in row]
    box_count = input_flat.count(box_color)
    distractor_count = sum(value not in (0, box_color) for value in input_flat)
    return box_color, box_count, distractor_count


def side_pair_dp() -> dict[str, Any]:
    """Necessary expanded-rectangle capacity bound for n=3..5 boxes."""
    pairs = [
        {"width": width, "height": height, "area": width * height, "expanded": (width + 1) * (height + 1)}
        for width in range(3, 11)
        for height in range(3, 11)
    ]
    result = {}
    for count in range(3, 6):
        dp: dict[int, tuple[int, list[dict[str, int]]]] = {0: (0, [])}
        for _ in range(count):
            next_dp: dict[int, tuple[int, list[dict[str, int]]]] = {}
            for used, (area, chosen) in dp.items():
                for pair in pairs:
                    expanded = used + pair["expanded"]
                    if expanded > 21 * 21:
                        continue
                    proposal = area + pair["area"]
                    if expanded not in next_dp or proposal > next_dp[expanded][0]:
                        next_dp[expanded] = (proposal, chosen + [pair])
            dp = next_dp
        used, (area, chosen) = max(dp.items(), key=lambda item: item[1][0])
        result[str(count)] = {
            "area_upper_bound": area,
            "expanded_area": used,
            "side_pair_certificate": chosen,
        }
    assert result["3"]["area_upper_bound"] == 300
    assert result["4"]["area_upper_bound"] == 361
    assert result["5"]["area_upper_bound"] == 355
    return result


def explicit_parameters() -> dict[str, dict[str, Any]]:
    # A=27, one isolated overwrite: B=26, D=1.  The box remains the unique
    # histogram argmax, but every target threshold suppresses it.
    false_negative = {
        "width": 10,
        "height": 10,
        "rows": [0],
        "cols": [0],
        "color": 1,
        "boxrows": [0, 0, 4],
        "boxcols": [0, 4, 0],
        "wides": [3, 3, 3],
        "talls": [3, 3, 3],
        "boxcolor": 2,
    }

    # A=B=48.  Choose 37 checkerboard cells below the boxes, so D=37 and
    # every target threshold selects both colors even though B>D.
    outside = [
        (row, col)
        for row in range(5, 20)
        for col in range(20)
        if (row + col) % 2 == 0
    ][:37]
    false_positive = {
        "width": 20,
        "height": 20,
        "rows": [row for row, _ in outside],
        "cols": [col for _, col in outside],
        "color": 1,
        "boxrows": [0, 0, 0],
        "boxcols": [0, 5, 10],
        "wides": [4, 4, 4],
        "talls": [4, 4, 4],
        "boxcolor": 2,
    }
    return {"false_negative": false_negative, "false_positive": false_positive}


def make_session(path: Path) -> onnxruntime.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(str(path))))
    assert model is not None
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(model.SerializeToString(), options)


def explicit_counterexamples() -> dict[str, Any]:
    params = explicit_parameters()
    examples = {name: generator.generate(**values) for name, values in params.items()}
    rows = {}
    for name, values in params.items():
        example = examples[name]
        _, box_count, distractor_count = count_colors(example)
        points = sorted(zip(values["rows"], values["cols"]))
        assert common.remove_neighbors(points) == points
        assert not common.overlaps(
            values["boxrows"], values["boxcols"], values["wides"], values["talls"], 1
        )
        rows[name] = {
            "parameters": values,
            "box_area": sum(w * h for w, h in zip(values["wides"], values["talls"])),
            "box_count_after_overwrite": box_count,
            "distractor_count_after_remove_neighbors": distractor_count,
            "box_is_unique_argmax": box_count > distractor_count,
            "random_pixels_exact_event_probability": (0.05 ** len(points))
            * (0.95 ** (values["width"] * values["height"] - len(points))),
        }
    assert rows["false_negative"]["box_count_after_overwrite"] == 26
    assert rows["false_negative"]["distractor_count_after_remove_neighbors"] == 1
    assert rows["false_positive"]["box_count_after_overwrite"] == 48
    assert rows["false_positive"]["distractor_count_after_remove_neighbors"] == 37

    candidate_checks = []
    for threshold in THRESHOLDS:
        path = (
            ROOT
            / "scripts/golf/loop_8004_42_plus20/root_task192_threshold_188/candidates"
            / f"task192_hardsigmoid_k{threshold}.onnx"
        )
        session = make_session(path)
        checks: dict[str, Any] = {}
        for name, example in examples.items():
            benchmark = scoring.convert_to_numpy(example)
            assert benchmark is not None
            raw = session.run(["output"], {"input": benchmark["input"]})[0]
            predicted = raw > 0.0
            expected = benchmark["output"].astype(bool)
            checks[name] = {
                "correct": bool(np.array_equal(predicted, expected)),
                "different_cells": int(np.count_nonzero(predicted != expected)),
                "runtime_error": False,
                "nonfinite_values": int(np.count_nonzero(~np.isfinite(raw))),
            }
        candidate_checks.append({"threshold": threshold, "checks": checks})
    assert all(
        not check["correct"]
        for row in candidate_checks
        for check in row["checks"].values()
    )
    return {"cases": rows, "candidate_checks": candidate_checks}


def wilson(failures: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = failures / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def quantile(counter: Counter[int], q: float) -> int:
    target = math.ceil(sum(counter.values()) * q)
    cumulative = 0
    for value in sorted(counter):
        cumulative += counter[value]
        if cumulative >= target:
            return value
    raise AssertionError("empty counter")


def sample_seed(seed: int, total: int) -> dict[str, Any]:
    random.seed(seed)
    box_hist: Counter[int] = Counter()
    distractor_hist: Counter[int] = Counter()
    thresholds = {
        threshold: {"fail": 0, "box_not_selected": 0, "distractor_selected": 0, "both_causes": 0}
        for threshold in THRESHOLDS
    }
    for _ in range(total):
        example = generator.generate()
        _, box_count, distractor_count = count_colors(example)
        box_hist[box_count] += 1
        distractor_hist[distractor_count] += 1
        for threshold, row in thresholds.items():
            low_box = box_count <= threshold
            high_distractor = distractor_count > threshold
            row["box_not_selected"] += int(low_box)
            row["distractor_selected"] += int(high_distractor)
            row["both_causes"] += int(low_box and high_distractor)
            row["fail"] += int(low_box or high_distractor)
    return {
        "seed": seed,
        "total": total,
        "box_count": {
            "minimum": min(box_hist),
            "maximum": max(box_hist),
            "q001": quantile(box_hist, 0.001),
            "q01": quantile(box_hist, 0.01),
            "q50": quantile(box_hist, 0.50),
        },
        "distractor_count": {
            "minimum": min(distractor_hist),
            "maximum": max(distractor_hist),
            "q50": quantile(distractor_hist, 0.50),
            "q99": quantile(distractor_hist, 0.99),
            "q999": quantile(distractor_hist, 0.999),
        },
        "thresholds": {
            str(threshold): {
                **row,
                "failure_rate": row["fail"] / total,
                "failure_rate_wilson95": wilson(row["fail"], total),
            }
            for threshold, row in thresholds.items()
        },
        "histograms": {
            "box": {str(key): value for key, value in sorted(box_hist.items())},
            "distractor": {str(key): value for key, value in sorted(distractor_hist.items())},
        },
    }


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["total"] for row in samples)
    result = {}
    for threshold in THRESHOLDS:
        source = [row["thresholds"][str(threshold)] for row in samples]
        values = {
            key: sum(int(row[key]) for row in source)
            for key in ("fail", "box_not_selected", "distractor_selected", "both_causes")
        }
        result[str(threshold)] = {
            **values,
            "total": total,
            "failure_rate": values["fail"] / total,
            "normal_accuracy_upper_bound": 1.0 - values["fail"] / total,
            "failure_rate_wilson95": wilson(values["fail"], total),
        }
    return {"total": total, "thresholds": result}


def main() -> None:
    root_before = {name: sha256(ROOT / name) for name in EXPECTED_ROOT}
    assert root_before == EXPECTED_ROOT
    stage_before = tree_digest(ROOT / "others" / "71407")

    dp = side_pair_dp()
    explicit = explicit_counterexamples()
    samples = [sample_seed(seed, SAMPLES_PER_SEED) for seed in SEEDS]
    combined = aggregate(samples)
    best_rate = min(row["failure_rate"] for row in combined["thresholds"].values())
    best = [
        int(threshold)
        for threshold, row in combined["thresholds"].items()
        if row["failure_rate"] == best_rate
    ]

    root_after = {name: sha256(ROOT / name) for name in EXPECTED_ROOT}
    stage_after = tree_digest(ROOT / "others" / "71407")
    assert root_after == root_before
    assert stage_after == stage_before
    result = {
        "task": 192,
        "generator": "task_7e0986d6",
        "decision": "NO_FIXED_THRESHOLD_IS_ALL_SUPPORT_EXACT",
        "formal_bounds": {
            "remove_neighbors": {
                "fixed_grid": "0 <= D <= ceil(width*height/2), both bounds tight",
                "global": {"minimum": 0, "maximum": 200, "tight": True},
                "proof": "output is a 4-neighbor independent set; every independent set is reachable by sampling exactly that set",
            },
            "box_area": {
                "global_minimum": 27,
                "global_maximum": 361,
                "tight": True,
                "expanded_rectangle_dp": dp,
                "maximum_witness": {
                    "width": 20,
                    "height": 20,
                    "boxrows": [0, 0, 11, 11],
                    "boxcols": [0, 11, 0, 11],
                    "wides": [10, 9, 10, 9],
                    "talls": [10, 10, 9, 9],
                    "area": 361,
                },
            },
            "box_color_after_overwrite": {
                "fixed_boxes": "sum(floor(width_i*height_i/2)) <= B <= sum(width_i*height_i)",
                "global": {"minimum": 12, "maximum": 361, "tight": True},
                "proof": "at most ceil(area_i/2) independent distractors can overwrite each spaced rectangle",
            },
            "selector_exact_condition": "B > k and D <= k",
            "all_support_impossibility": "would require k < 12 and k >= 200 simultaneously",
        },
        "explicit_reachable_counterexamples": explicit,
        "monte_carlo": {
            "seeds": list(SEEDS),
            "per_seed": SAMPLES_PER_SEED,
            "samples": samples,
            "aggregate": combined,
            "empirical_best_thresholds": best,
            "empirical_best_failure_rate": best_rate,
            "scope": "generator distribution estimate, not an all-support guarantee",
        },
        "policy": {
            "guaranteed": False,
            "policy100_supported": False,
            "policy90_requires_empirical_accuracy_at_least": 0.90,
        },
        "guards": {
            "root_before": root_before,
            "root_after": root_after,
            "others_71407_before": stage_before,
            "others_71407_after": stage_after,
        },
        "root_mutations": [],
    }
    (HERE / "audit" / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "best_thresholds": best,
                "best_failure_rate": best_rate,
                "aggregate": combined["thresholds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
