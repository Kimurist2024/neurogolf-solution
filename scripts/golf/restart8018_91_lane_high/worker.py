#!/usr/bin/env python3
"""Urgent evidence-only cost-400..500 search on the 8018.91 authority.

The lane rebases every matching historical loose ONNX and ZIP member, tries
exact graph simplifications and low-cost templates, and admits a candidate only
after strict structure, local+official gold, stable margin and two independent
fresh-2000 streams at 100%.  It never writes root authority files.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = import_path(
    "restart8018_91_high_base",
    ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py",
)
TRY = import_path(
    "restart8018_91_high_try",
    ROOT / "scripts/golf/try_candidate.py",
)
VERIFY = import_path(
    "restart8018_91_high_verify",
    ROOT / "scripts/verify_fix.py",
)

AUTHORITY = ROOT / "submission_base_8018.91.zip"
AUTHORITY_SHA256 = "e43865760ec8807fbb217fba718226ca6b86d9128b911479214e3252b9f9e091"

# Current measured costs from the immutable 8018.91 authority.  Ordered by the
# user's priority list; round-robin workers preserve disjoint ownership.
BAND: tuple[tuple[int, int], ...] = (
    (102, 493),
    (156, 483),
    (374, 481),
    (25, 472),
    (250, 468),
    (270, 465),
    (62, 442),
    (324, 434),
    (275, 428),
    (308, 427),
)

# Do not silently suppress requested tasks.  Known-black history is evidence,
# not admission: any candidate still has to pass both local and official gold.
PRIVATE_ZERO_OR_UNSOUND: set[int] = set()
ELIGIBLE = tuple(task for task, _cost in BAND)
COSTS = dict(BAND)

BASE.HERE = HERE
BASE.AUTHORITY = AUTHORITY
BASE.AUTHORITY_SHA256 = AUTHORITY_SHA256
BASE.BAND = BAND
BASE.PRIVATE_ZERO_OR_UNSOUND = PRIVATE_ZERO_OR_UNSOUND
BASE.EXPLICIT_LATEST_LB_BLACK = set()
BASE.ELIGIBLE = ELIGIBLE
BASE.COSTS = COSTS
BASE.CHANGED_FROM_8011_05 = set(ELIGIBLE)
BASE.THRESHOLD = 1.0
BASE.FRESH_PER_SEED = 2_000
BASE.SUPPORT.POLICY_THRESHOLD = 1.0
BASE.SUPPORT.FRESH_PER_SEED = 2_000


def exact_failfast_evaluate_four(
    data: bytes, cases: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Exact gate: one mismatch/error/margin fault is conclusively fatal."""
    return {"disable_threads1": BASE.failfast_known(data, cases)}


BASE.SUPPORT.evaluate_four = exact_failfast_evaluate_four


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_gate(path: Path, task: int, authority_cost: int) -> dict[str, Any]:
    """Non-promoting try_candidate gate plus the independent official module."""
    stream = io.StringIO()
    result: dict[str, Any] = {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "authority_cost": authority_cost,
    }
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        file_ok = TRY._validate_file_size(path)
        model = onnx.load(str(path)) if file_ok else None
        structure_ok = bool(model is not None and TRY._validate_ops_and_shapes(model))
        local_gold = False
        mismatch = None
        margin_ok = False
        minimum_positive = None
        score = None
        if structure_ok and model is not None:
            local_gold, mismatch = TRY._verify_gold(model, task)
        official_gold = bool(local_gold and VERIFY.official_gold(path, task))
        if official_gold and model is not None:
            margin_ok, minimum_positive = TRY._check_margin(model, task)
        if margin_ok and model is not None:
            with tempfile.TemporaryDirectory(prefix="high8018_gate_") as workdir:
                score = TRY._score_model(
                    model, task, workdir, "candidate", require_correct=True
                )
    result.update(
        {
            "file_ok": file_ok,
            "structure_ok": structure_ok,
            "local_gold_exact": local_gold,
            "official_gold_exact": official_gold,
            "margin_ok": margin_ok,
            "minimum_positive": minimum_positive,
            "scored": score is not None,
            "candidate_cost": None if score is None else int(score.cost),
            "strictly_cheaper": bool(score is not None and score.cost < authority_cost),
            "pass": bool(
                file_ok
                and structure_ok
                and local_gold
                and official_gold
                and margin_ok
                and score is not None
                and score.cost < authority_cost
            ),
            "first_mismatch": None
            if mismatch is None
            else {"subset": mismatch.subset, "index": mismatch.index},
            "gate_log": stream.getvalue(),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if digest(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("8018.91 authority SHA mismatch")

    payload = BASE.Worker(args.worker).run()
    strict_gates: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for row in payload["finalists"]:
        path = ROOT / row["saved_path"]
        gate = strict_gate(path, int(row["task"]), int(row["authority_cost"]))
        strict_gates.append(gate)
        if gate["pass"]:
            row["strict_gate"] = gate
            accepted.append(row)

    payload["pre_strict_gate_finalists"] = payload["finalists"]
    payload["strict_gates"] = strict_gates
    payload["finalists"] = accepted
    payload["threshold"] = 1.0
    payload["fresh_per_seed"] = 2_000
    payload["absolute_admission_gate"] = (
        "strict static structure + local gold exact + official gold exact + "
        "stable margin + two independent fresh-2000 streams at 100%"
    )
    payload["protected_writes"] = (
        "scripts/golf/restart8018_91_lane_high only; root authority untouched"
    )
    output = HERE / f"worker_{args.worker}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "worker": args.worker,
                "tasks": payload["assigned_tasks"],
                "strict_winners": [
                    {
                        "task": row["task"],
                        "cost": row["candidate_cost"],
                        "gain": row["score_gain"],
                        "path": row["saved_path"],
                    }
                    for row in accepted
                ],
                "elapsed": payload["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
