#!/usr/bin/env python3
"""Fail-closed final audit for the exact task344 cost-132 reparameterization."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import importlib.util
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
BASELINE = HERE / "baseline/task344.onnx"
CANDIDATE = HERE / "candidates/task344_compact_g_cost132.onnx"
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
ROOT_SUBMISSION = ROOT / "submission.zip"
ROOT_SCORES = ROOT / "all_scores.csv"
EXPECTED_AUTHORITY_ZIP = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
EXPECTED_AUTHORITY_MEMBER = "05bedf3ca834aadfc973c00fc91cafdb4d0ae1aaab374115d924e2e33fb1bf6c"
EXPECTED_ROOT_SCORES = "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78"
FRESH_SEEDS = (344171501, 344171777)
FRESH_COUNT = 10_000

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import structure  # noqa: E402
from scripts.lib import scoring  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rule(grid: list[list[int]]) -> list[list[int]]:
    result = [row[:] for row in grid]
    height, width = len(grid), len(grid[0])
    for row in range(height):
        for col in range(width):
            if grid[row][col] != 3:
                continue
            touched = False
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < height and 0 <= cc < width and grid[rr][cc] == 2:
                    result[rr][cc] = 0
                    touched = True
            if touched:
                result[row][col] = 8
    return result


def arrays(path: Path) -> dict[str, np.ndarray]:
    return {
        item.name: numpy_helper.to_array(item).astype(np.float64)
        for item in onnx.load(path).graph.initializer
    }


def example_states(example: dict, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = example["input"]
    height, width = len(grid), len(grid[0])
    x = np.zeros((10, height, width), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            x[grid[row][col], row, col] = 1.0
    z = np.einsum(
        "dpq,ph,qw->dhw",
        x,
        kernel[:height, :height],
        kernel[:width, :width],
        optimize=True,
    )
    centers = np.asarray(grid, dtype=np.int64).reshape(-1)
    targets = np.asarray(example["output"], dtype=np.int64).reshape(-1)
    return z.transpose(1, 2, 0).reshape(-1, 10), centers, targets


def direct_logits(
    z: np.ndarray,
    center: np.ndarray,
    authority: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    v = authority["V"]
    u = z @ v.T
    current = ((u @ authority["H"].T) @ authority["S"] @ authority["H"])
    current *= v[:, center].T
    current = current @ authority["M"] @ v

    cv = candidate["V"]
    compact = u @ candidate["G"]
    compact *= cv[:, center].T
    compact = compact @ candidate["M"] @ cv
    return current, compact


def direct_stream(examples: list[dict], authority: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict:
    b = authority["B"]
    kernel = np.power(b.T @ b, 16)
    right = wrong = wrong_cells = nonfinite = rule_wrong = 0
    authority_right = authority_wrong = authority_candidate_sign_mismatch = 0
    max_abs_delta = 0.0
    min_positive = float("inf")
    max_negative = -float("inf")
    min_abs = float("inf")
    first_failure = None
    for index, example in enumerate(examples):
        z, center, target = example_states(example, kernel)
        raw_authority, raw_candidate = direct_logits(z, center, authority, candidate)
        expected = np.eye(10, dtype=bool)[target]
        predicted = raw_candidate > 0
        authority_predicted = raw_authority > 0
        failures = int(np.count_nonzero(predicted != expected))
        wrong_cells += failures
        if failures:
            wrong += 1
            if first_failure is None:
                first_failure = {"index": index, "wrong_cells": failures}
        else:
            right += 1
        if np.array_equal(authority_predicted, expected):
            authority_right += 1
        else:
            authority_wrong += 1
        authority_candidate_sign_mismatch += int(np.count_nonzero(authority_predicted != predicted))
        rule_wrong += int(rule(example["input"]) != example["output"])
        nonfinite += int(np.count_nonzero(~np.isfinite(raw_candidate)))
        max_abs_delta = max(max_abs_delta, float(np.max(np.abs(raw_candidate - raw_authority))))
        positive = raw_candidate[expected]
        negative = raw_candidate[~expected]
        min_positive = min(min_positive, float(positive.min()))
        max_negative = max(max_negative, float(negative.max()))
        nonzero = np.abs(raw_candidate)[np.abs(raw_candidate) > 0]
        if nonzero.size:
            min_abs = min(min_abs, float(nonzero.min()))
    return {
        "right": right,
        "wrong": wrong,
        "total": right + wrong,
        "wrong_cells": wrong_cells,
        "authority_right": authority_right,
        "authority_wrong": authority_wrong,
        "authority_candidate_sign_mismatch_cells": authority_candidate_sign_mismatch,
        "rule_wrong": rule_wrong,
        "nonfinite": nonfinite,
        "min_positive": min_positive,
        "max_negative": max_negative,
        "min_abs_nonzero": min_abs,
        "max_abs_raw_delta_vs_authority": max_abs_delta,
        "first_failure": first_failure,
    }


def ort_known(model: onnx.ModelProto, mode: str, level: ort.GraphOptimizationLevel) -> dict:
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    session = ort.InferenceSession(sanitized.SerializeToString(), options)
    examples = scoring.load_examples(344)
    right = wrong = errors = nonfinite = near = 0
    min_abs = float("inf")
    shapes: set[tuple[int, ...]] = set()
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            bench = scoring.convert_to_numpy(example)
            try:
                raw = session.run(["output"], {"input": bench["input"]})[0]
                shapes.add(tuple(int(item) for item in raw.shape))
                nonfinite += int(np.count_nonzero(~np.isfinite(raw)))
                nonzero = np.abs(raw)[np.abs(raw) > 0]
                if nonzero.size:
                    min_abs = min(min_abs, float(nonzero.min()))
                near += int(np.count_nonzero((np.abs(raw) > 0) & (np.abs(raw) < 0.25)))
                if np.array_equal(raw > 0, bench["output"] > 0):
                    right += 1
                else:
                    wrong += 1
            except Exception:
                errors += 1
    return {
        "mode": mode,
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "total": right + wrong + errors,
        "nonfinite": nonfinite,
        "near_abs_lt_0_25": near,
        "min_abs_nonzero": min_abs,
        "runtime_shapes": [list(item) for item in sorted(shapes)],
    }


def team_audit() -> dict:
    path = ROOT / "others/3/7907/neurogolf_team_validator_v1/ngolf_validator.py"
    spec = importlib.util.spec_from_file_location("task344_rebase171_team_validator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    audit, failures = module.audit_model_bytes(
        CANDIDATE.read_bytes(),
        344,
        ROOT / "inputs/neurogolf-2026",
        source="task344_rebase171_compact_g132",
        trace_dir=HERE / "traces",
    )
    return {"audit": dataclasses.asdict(audit), "failures": failures}


def main() -> None:
    if sha256(AUTHORITY_ZIP) != EXPECTED_AUTHORITY_ZIP or sha256(ROOT_SUBMISSION) != EXPECTED_AUTHORITY_ZIP:
        raise RuntimeError("8009.46 authority archive changed")
    if sha256(BASELINE) != EXPECTED_AUTHORITY_MEMBER:
        raise RuntimeError("task344 authority member changed")
    if sha256(ROOT_SCORES) != EXPECTED_ROOT_SCORES:
        raise RuntimeError("all_scores.csv changed")

    authority_arrays = arrays(BASELINE)
    candidate_arrays = arrays(CANDIDATE)
    local = authority_arrays["H"].T @ authority_arrays["S"] @ authority_arrays["H"]
    identity_residual = candidate_arrays["G"] - local
    colors = [0, 2, 3, 5]
    authority_tensor = np.einsum(
        "sl,ld,kc,sx,xk,ky,yo->dco",
        authority_arrays["H"],
        authority_arrays["V"],
        authority_arrays["V"],
        authority_arrays["S"],
        authority_arrays["H"],
        authority_arrays["M"],
        authority_arrays["V"],
        optimize=True,
    )
    candidate_tensor = np.einsum(
        "ld,kc,lk,ky,yo->dco",
        candidate_arrays["V"],
        candidate_arrays["V"],
        candidate_arrays["G"],
        candidate_arrays["M"],
        candidate_arrays["V"],
        optimize=True,
    )
    coefficient_error = candidate_tensor - authority_tensor
    kernel = np.power(authority_arrays["B"].T @ authority_arrays["B"], 16)
    max_axis_weight = max(float(kernel[:size, position].sum()) for size in range(3, 11) for position in range(size))
    max_weight_sum = max_axis_weight * max_axis_weight
    max_coefficient_error = float(np.max(np.abs(coefficient_error[np.ix_(colors, colors, range(10))])))
    global_logit_error_bound = max_weight_sum * max_coefficient_error
    known_payload = scoring.load_examples(344)
    known_examples = known_payload["train"] + known_payload["test"] + known_payload["arc-gen"]
    direct_known = direct_stream(known_examples, authority_arrays, candidate_arrays)

    generator = importlib.import_module("task_d90796e8")
    fresh = []
    for seed in FRESH_SEEDS:
        random.seed(seed)
        examples = [generator.generate() for _ in range(FRESH_COUNT)]
        row = direct_stream(examples, authority_arrays, candidate_arrays)
        row["seed"] = seed
        fresh.append(row)

    model = onnx.load(CANDIDATE)
    modes = [
        ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
        ("extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
        ("default_enable_all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ]
    known4 = [ort_known(model, name, level) for name, level in modes]
    official = scoring.score_and_verify(
        copy.deepcopy(model),
        344,
        str(HERE / "traces"),
        label="task344_rebase171_compact_g132",
    )
    team = team_audit()
    structural = structure(copy.deepcopy(model), 344)
    cost = cost_of(str(CANDIDATE))
    build = json.loads((HERE / "audit/exact_build.json").read_text())
    mechanical_probe_gate = (
        cost == (0, 132, 132)
        and official is not None
        and official["correct"]
        and all(row["right"] == 266 and row["wrong"] == 0 and row["errors"] == 0 and row["nonfinite"] == 0 for row in known4)
        and direct_known["wrong"] == 0
        and direct_known["authority_candidate_sign_mismatch_cells"] == 0
        and all(row["authority_candidate_sign_mismatch_cells"] == 0 and row["nonfinite"] == 0 for row in fresh)
        and team["audit"]["valid"]
        and structural["checker_full"]
        and structural["strict_data_prop"]
        and structural["runtime_shapes"]["mismatch_count"] == 0
        and not structural["banned_ops"]
        and not structural["lookup_or_scatter"]
        and not structural["conv_bias_findings"]
    )
    result = {
        "lane": "agent_task344_rebase_171",
        "task": 344,
        "authority": {
            "lb": 8009.46,
            "archive": "submission_base_8009.46.zip",
            "archive_sha256": EXPECTED_AUTHORITY_ZIP,
            "member_sha256": EXPECTED_AUTHORITY_MEMBER,
            "generator": "inputs/arc-gen-repo/tasks/task_d90796e8.py",
            "generator_hash8": "d90796e8",
            "cost": {"memory": 0, "params": 137, "cost": 137},
            "private_zero_history": False,
            "lb_white": True,
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": sha256(CANDIDATE),
            "serialized_bytes": CANDIDATE.stat().st_size,
            "cost": {"memory": cost[0], "params": cost[1], "cost": cost[2]},
            "strict_lower": cost[2] < 137,
            "score": official["score"] if official else None,
            "gain": (official["score"] - (25.0 - np.log(137.0))) if official else None,
        },
        "proof": {
            "rule": "Simultaneously: 3 with cardinally adjacent 2 becomes 8; each touched 2 becomes 0; otherwise copy.",
            "finite_support": "Generator dimensions are 3..10 and colors are {0,2,3,5}; B is exactly zero after coordinate 9. However random_pixels makes every gray subset have nonzero support, so fresh sampling is not a complete finite-support enumeration.",
            "identity_target": "G_target = H.T @ S @ H",
            "identity_max_abs_residual": float(np.max(np.abs(identity_residual))),
            "max_supported_color_coefficient_error": max_coefficient_error,
            "max_spatial_weight_sum_dims_3_to_10": max_weight_sum,
            "global_real_logit_error_upper_bound": global_logit_error_bound,
            "authority_minimum_sign_margin_full_support": None,
            "authority_margin_proof_status": "NOT_PROVED: the independent Bernoulli gray subset alone has up to 2^(H*W) reachable states; no exhaustive margin certificate was established.",
            "lookup_or_fixture_used": False,
        },
        "official_score_and_verify": official,
        "team_validator": team,
        "structure": structural,
        "known4": known4,
        "direct_known": direct_known,
        "fresh_2x10000_serialized_weight_evaluator": fresh,
        "rejected_exact_sparse_B": {
            "decision": build["sparse_decision"],
            "reason": build["sparse_full_checker_error"],
        },
        "mechanical_probe_gate": mechanical_probe_gate,
        "winner_eligible": False,
        "probe_only": mechanical_probe_gate,
        "verdict": "PROBE_ONLY" if mechanical_probe_gate else "REJECT",
        "protected_files_modified": False,
    }
    (HERE / "audit/final_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "result.json").write_text(
        json.dumps(
            {
                "lane": result["lane"],
                "verdict": result["verdict"],
                "winner": None,
                "probe_candidate": result["candidate"] if mechanical_probe_gate else None,
                "authority_cost": 137,
                "winner_cost": None,
                "potential_score_gain": result["candidate"]["gain"] if mechanical_probe_gate else 0.0,
                "root_modified": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"verdict": result["verdict"], "candidate": result["candidate"], "known4": known4, "fresh": fresh}, indent=2))


if __name__ == "__main__":
    main()
