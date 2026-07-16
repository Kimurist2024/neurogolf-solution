#!/usr/bin/env python3
"""Fail-closed audit for the exact task192 polynomial candidate.

This script is deliberately non-promoting: it only reads the immutable 8006.61
authority and writes evidence below this lane.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATE = HERE / "candidates/task192_exact_poly.onnx"
AUTHORITY = ROOT / "others/71403/lb_verified_8006.61/submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
BASELINE_MEMBER_SHA256 = "e7f9a11b93b611acfa4bba39e90e1ddf24223d50add4277fe9716f21f6ede10c"
SEEDS = (192800661, 192930007)
FRESH_PER_SEED = 5000

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STATIC = load_module(
    "sound93_static",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_sweep_wave30b/audit_sweep.py",
)
KNOWN = load_module(
    "sound93_known",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20b_86/audit_actual_lower.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def task192_rule(grid: list[list[int]]) -> list[list[int]]:
    counts = [sum(row.count(color) for row in grid) for color in range(10)]
    selected = max(range(1, 10), key=lambda color: counts[color])
    height, width = len(grid), len(grid[0])
    output = [[0] * width for _ in range(height)]
    for row in range(height):
        for col in range(width):
            horizontal = any(
                grid[row][other] == selected
                for other in range(max(0, col - 1), min(width, col + 2))
            )
            vertical = any(
                grid[other][col] == selected
                for other in range(max(0, row - 1), min(height, row + 2))
            )
            if grid[row][col] != 0 and horizontal and vertical:
                output[row][col] = selected
    return output


def make_session(data: bytes) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model returned None")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def fresh_audit(data: bytes, seed: int, count: int) -> dict[str, Any]:
    generator = importlib.import_module("task_7e0986d6")
    random.seed(seed)
    session = make_session(data)
    reference_right = 0
    model_right = 0
    runtime_errors = 0
    nonfinite_values = 0
    min_positive: float | None = None
    max_nonpositive = -math.inf
    first_failure: dict[str, Any] | None = None
    for index in range(count):
        example = generator.generate()
        reference_right += int(task192_rule(example["input"]) == example["output"])
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            runtime_errors += 1
            first_failure = first_failure or {"index": index, "error": "convert_to_numpy returned None"}
            continue
        try:
            raw = np.asarray(
                session.run(
                    [session.get_outputs()[0].name],
                    {session.get_inputs()[0].name: benchmark["input"]},
                )[0]
            )
        except Exception as exc:  # noqa: BLE001
            runtime_errors += 1
            first_failure = first_failure or {"index": index, "error": f"{type(exc).__name__}: {exc}"}
            continue
        finite = np.isfinite(raw)
        nonfinite_values += int(raw.size - np.count_nonzero(finite))
        expected = benchmark["output"].astype(bool)
        correct = np.array_equal(raw > 0, expected)
        model_right += int(correct)
        if not correct and first_failure is None:
            first_failure = {
                "index": index,
                "different_cells": int(np.count_nonzero((raw > 0) != expected)),
                "height": len(example["input"]),
                "width": len(example["input"][0]),
            }
        positives = raw[expected]
        negatives = raw[~expected]
        if positives.size:
            value = float(positives.min())
            min_positive = value if min_positive is None else min(min_positive, value)
        if negatives.size:
            max_nonpositive = max(max_nonpositive, float(negatives.max()))
    return {
        "seed": seed,
        "total": count,
        "reference_right": reference_right,
        "model_right": model_right,
        "runtime_errors": runtime_errors,
        "nonfinite_values": nonfinite_values,
        "min_positive": min_positive,
        "max_nonpositive": max_nonpositive,
        "first_failure": first_failure,
        "perfect": (
            reference_right == count
            and model_right == count
            and runtime_errors == 0
            and nonfinite_values == 0
        ),
    }


def exhaustive_local_sign_proof() -> dict[str, Any]:
    """Exhaust every possible local count tuple used by the exact formula."""
    cases = 0
    failures: list[dict[str, int]] = []
    # Outside the logical grid, center masks are zero and every output is zero.
    cases += 1
    for horizontal_inside in (1, 2, 3):
        for vertical_inside in (1, 2, 3):
            background_product = horizontal_inside * vertical_inside
            for center_nonzero in (0, 1):
                for horizontal_a in range(horizontal_inside + 1):
                    for vertical_a in range(vertical_inside + 1):
                        cases += 1
                        product = center_nonzero * horizontal_a * vertical_a
                        selected_raw = product
                        background_raw = background_product - 9 * product
                        expected_selected = bool(
                            center_nonzero and horizontal_a > 0 and vertical_a > 0
                        )
                        correct = (
                            (selected_raw > 0) == expected_selected
                            and (background_raw > 0) == (not expected_selected)
                        )
                        if not correct:
                            failures.append(
                                {
                                    "horizontal_inside": horizontal_inside,
                                    "vertical_inside": vertical_inside,
                                    "center_nonzero": center_nonzero,
                                    "horizontal_a": horizontal_a,
                                    "vertical_a": vertical_a,
                                    "selected_raw": selected_raw,
                                    "background_raw": background_raw,
                                }
                            )
    return {
        "cases": cases,
        "failures": failures,
        "perfect": not failures,
        "argument": (
            "P=center_nonzero*horizontal_A*vertical_A. P>0 iff all three rule "
            "conditions hold. Inside-grid B=horizontal_inside*vertical_inside is "
            "in [1,9]. Therefore B-9P>0 iff P=0; when P>=1 it is <=0. "
            "Outside-grid the center factors are zero, so all outputs are zero."
        ),
    }


def main() -> int:
    authority_data = AUTHORITY.read_bytes()
    if digest(authority_data) != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority ZIP hash changed")
    with zipfile.ZipFile(AUTHORITY) as archive:
        baseline_data = archive.read("task192.onnx")
    if digest(baseline_data) != BASELINE_MEMBER_SHA256:
        raise RuntimeError("authority task192 member hash changed")
    candidate_data = CANDIDATE.read_bytes()

    baseline_profile = STATIC.profiler_cost(baseline_data, 192, "authority_task192")
    candidate_profile = STATIC.profiler_cost(candidate_data, 192, "candidate_task192_exact_poly")
    static = STATIC.static_audit(candidate_data)
    known_four = {
        label: KNOWN.known_config(192, candidate_data, disable, threads)
        for disable, threads, label in KNOWN.CONFIGS
    }
    trace = KNOWN.direct_runtime_shape_trace(192, candidate_data)
    fresh = [fresh_audit(candidate_data, seed, FRESH_PER_SEED) for seed in SEEDS]
    proof = exhaustive_local_sign_proof()
    reasons: list[str] = []
    if static["reasons"]:
        reasons.append("static_gate")
    if candidate_profile["cost"] >= baseline_profile["cost"]:
        reasons.append("not_strictly_cheaper")
    if not all(row.get("perfect", False) for row in known_four.values()):
        reasons.append("known_four_not_perfect")
    if not trace.get("truthful", False):
        reasons.append("runtime_shape_not_truthful")
    if not all(row["perfect"] for row in fresh):
        reasons.append("fresh_not_perfect")
    if not proof["perfect"]:
        reasons.append("local_sign_proof_failed")

    gain = math.log(baseline_profile["cost"] / candidate_profile["cost"])
    report = {
        "authority": {
            "path": str(AUTHORITY.relative_to(ROOT)),
            "zip_sha256": AUTHORITY_SHA256,
            "task192_sha256": BASELINE_MEMBER_SHA256,
            "profile": baseline_profile,
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": digest(candidate_data),
            "serialized_bytes": len(candidate_data),
            "profile": candidate_profile,
            "cost_reduction": baseline_profile["cost"] - candidate_profile["cost"],
            "projected_gain": gain,
        },
        "static": static,
        "known_four_configs": known_four,
        "runtime_shape_trace": trace,
        "fresh": fresh,
        "exhaustive_local_sign_proof": proof,
        "reasons": reasons,
        "accepted": not reasons,
    }
    (HERE / "audit").mkdir(exist_ok=True)
    output = HERE / "audit/task192_exact_poly.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"task192 baseline={baseline_profile['cost']} candidate={candidate_profile['cost']} "
        f"gain={gain:.12f} known4={all(row.get('perfect', False) for row in known_four.values())} "
        f"fresh={[row['model_right'] for row in fresh]} accepted={not reasons}",
        flush=True,
    )
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
