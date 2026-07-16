#!/usr/bin/env python3
"""Two-seed/four-ORT-configuration POLICY95 audit for one historical model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx
import onnxruntime
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
PRIVATE_ZERO = {
    9, 15, 18, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 93, 96,
    101, 102, 112, 118, 133, 134, 138, 145, 157, 158, 168, 169,
    170, 173, 174, 178, 185, 187, 192, 196, 198, 202, 205, 208,
    209, 216, 219, 222, 233, 246, 251, 255, 273, 277, 285, 286,
    302, 319, 325, 333, 343, 346, 361, 365, 366, 372, 377, 379,
    391, 393, 396,
}


def load_base():
    path = ROOT / "scripts/golf/half_cost_51_100_303/audit_task070_policy95.py"
    spec = importlib.util.spec_from_file_location("policy95_307_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def evaluate(base, runtime, cases):
    """Evaluate fail-closed when a malformed graph returns a noncanonical shape."""
    right = wrong = errors = nonfinite = shape = small = 0
    minimum_positive = math.inf
    first_wrong = first_error = None
    for index, benchmark in enumerate(cases):
        try:
            raw = base.scoring._raw_output(runtime, benchmark["input"])
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_error is None:
                first_error = {"index": index, "type": type(exc).__name__,
                               "message": str(exc)}
            continue
        if tuple(raw.shape) != (1, 10, 30, 30):
            shape += 1
            wrong += 1
            if first_wrong is None:
                first_wrong = {"index": index, "output_shape": list(raw.shape)}
            continue
        if not np.all(np.isfinite(raw)):
            nonfinite += 1
            continue
        positives = raw[raw > 0]
        if positives.size:
            minimum_positive = min(minimum_positive, float(positives.min()))
            small += int(np.count_nonzero((positives > 0) & (positives < 0.25)))
        predicted = (raw > 0).astype(np.float32)
        if np.array_equal(predicted, benchmark["output"]):
            right += 1
        else:
            wrong += 1
            if first_wrong is None:
                first_wrong = {"index": index,
                               "mismatch_cells": int(np.count_nonzero(
                                   predicted != benchmark["output"]))}
    total = right + wrong + errors + nonfinite
    return {"total": total, "right": right, "wrong": wrong, "errors": errors,
            "nonfinite_cases": nonfinite, "shape_mismatches": shape,
            "small_positive_elements_0_to_0_25": small,
            "minimum_positive": None if minimum_positive is math.inf else minimum_positive,
            "accuracy": right / total if total else None,
            "first_wrong": first_wrong, "first_error": first_error}


def main() -> int:
    onnxruntime.set_default_logger_severity(3)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--fresh", type=int, default=2000)
    args = parser.parse_args()
    candidate = args.onnx.resolve()
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)

    base = load_base()
    base.ROOT = ROOT
    base.HERE = HERE
    base.AUTHORITY_ZIP = AUTHORITY
    base.AUTHORITY_ZIP_SHA256 = AUTHORITY_SHA256
    base.TASK = args.task
    base.SEEDS = (307_000_000 + args.task * 10 + 1, 307_000_000 + args.task * 10 + 2)
    base.FRESH_PER_SEED = args.fresh
    base.POLICY_RATE = 0.95

    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_model = onnx.load_model_from_string(
            archive.read(f"task{args.task:03d}.onnx")
        )
    candidate_model = onnx.load(candidate)
    with tempfile.TemporaryDirectory(prefix=f"policy95_{args.task:03d}_", dir="/tmp") as tmp:
        authority_profile = base.scoring.score_and_verify(
            authority_model, args.task, tmp, label="authority", require_correct=False
        )
        candidate_profile = base.scoring.score_and_verify(
            candidate_model, args.task, tmp, label="candidate", require_correct=False
        )
    stable, margin = base.scoring.model_margin_stable(candidate_model, args.task)
    known = base.cases_known()
    fresh_sets = {}
    fresh_attempts = {}
    for seed in base.SEEDS:
        fresh_sets[seed], fresh_attempts[seed] = base.cases_fresh(seed)

    configurations = [(mode, threads) for mode in ("disabled", "default") for threads in (1, 4)]
    known_rows = []
    fresh_rows = []
    for mode, threads in configurations:
        try:
            runtime = base.session(candidate_model, mode, threads)
        except Exception as exc:  # noqa: BLE001
            fail = {
                "optimization": mode,
                "threads": threads,
                "session_error": f"{type(exc).__name__}:{exc}",
            }
            known_rows.append(fail)
            for seed in base.SEEDS:
                fresh_rows.append({**fail, "seed": seed})
            continue
        known_rows.append({
            "optimization": mode, "threads": threads,
            **evaluate(base, runtime, known),
        })
        for seed in base.SEEDS:
            fresh_rows.append({
                "optimization": mode,
                "threads": threads,
                "seed": seed,
                "generation_attempts": fresh_attempts[seed],
                **evaluate(base, runtime, fresh_sets[seed]),
            })

    structural = base.structure(candidate_model)
    known_pass = all(
        not row.get("session_error")
        and row["wrong"] == row["errors"] == row["nonfinite_cases"] == 0
        and row["shape_mismatches"] == row["small_positive_elements_0_to_0_25"] == 0
        for row in known_rows
    )
    fresh_pass = all(
        not row.get("session_error")
        and row["errors"] == row["nonfinite_cases"] == 0
        and row["shape_mismatches"] == row["small_positive_elements_0_to_0_25"] == 0
        and row["accuracy"] >= 0.95
        for row in fresh_rows
    )
    actual_lower = bool(
        authority_profile and candidate_profile
        and int(candidate_profile["cost"]) < int(authority_profile["cost"])
    )
    admit = bool(structural["safe"] and known_pass and fresh_pass and actual_lower and stable)
    private = args.task in PRIVATE_ZERO
    classification = (
        "POLICY95_PRIVATE_ZERO_RISK" if admit and private else
        "POLICY95_UNPROVEN" if admit else
        "REJECT"
    )
    payload = {
        "task": args.task,
        "authority": {
            "zip": AUTHORITY.name,
            "zip_sha256": AUTHORITY_SHA256,
            "model_sha256": sha256(authority_model.SerializeToString()),
            "profile": authority_profile,
        },
        "candidate": {
            "path": str(candidate.relative_to(ROOT)),
            "sha256": sha256(candidate.read_bytes()),
            "profile": candidate_profile,
        },
        "cost_delta": None if not (authority_profile and candidate_profile)
        else int(authority_profile["cost"]) - int(candidate_profile["cost"]),
        "score_gain": None if not (authority_profile and candidate_profile)
        else math.log(authority_profile["cost"] / candidate_profile["cost"]),
        "structure": structural,
        "margin_stable": bool(stable),
        "margin_min": margin,
        "known_case_count": len(known),
        "known_four_configs": known_rows,
        "fresh_policy_rate": 0.95,
        "fresh_seeds": list(base.SEEDS),
        "fresh_per_seed": args.fresh,
        "fresh_four_configs_two_seeds": fresh_rows,
        "known_pass": known_pass,
        "fresh_pass": fresh_pass,
        "actual_strict_lower": actual_lower,
        "known_private_zero_lineage": private,
        "classification": classification,
        "admit_policy95": admit,
        "guaranteed_safe": False,
        "protected_writes": "none; evidence directory only",
    }
    out = HERE / f"task{args.task:03d}_{args.label}_policy95_audit.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "task": args.task,
        "classification": classification,
        "cost_delta": payload["cost_delta"],
        "score_gain": payload["score_gain"],
        "known_pass": known_pass,
        "fresh_pass": fresh_pass,
        "fresh_accuracies": sorted({row.get("accuracy") for row in fresh_rows
                                    if row.get("accuracy") is not None}),
        "evidence": str(out.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
