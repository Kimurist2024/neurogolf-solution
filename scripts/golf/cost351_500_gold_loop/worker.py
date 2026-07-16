#!/usr/bin/env python3
"""Gold-exact, evidence-only worker for the 8012.23 cost-351..500 lane.

This reuses the mature archive/simplifier/template scanner from wave 408, but
raises every accuracy gate to 100% and then independently calls the official
``try_candidate.py`` validation functions.  Root submission and score ledgers
are never modified.
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
    "cost351_500_wave408_base",
    ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py",
)
TRY = import_path(
    "cost351_500_try_candidate",
    ROOT / "scripts/golf/try_candidate.py",
)

AUTHORITY = ROOT / "submission_base_8012.23.zip"
AUTHORITY_SHA256 = "720ebf75d826945250e3c7d7ea11780a950d8d3038546e9c7595503277a1189f"

BAND: tuple[tuple[int, int], ...] = (
    (13, 352), (92, 357), (297, 361), (234, 361),
    (226, 370), (48, 379), (222, 380), (170, 382), (245, 384),
    (239, 384), (99, 389), (345, 389), (160, 396), (279, 397),
    (338, 403), (109, 405), (377, 409), (168, 414), (112, 419),
    (268, 420), (333, 421), (184, 421), (134, 422), (275, 428),
    (8, 431), (308, 433), (324, 435), (310, 448), (354, 461),
    (62, 462), (250, 468), (25, 474), (374, 481), (270, 489),
    (156, 493), (102, 493),
)

# The ledger still says task014=356, but the immutable 8012.23 ZIP profiles at
# cost 288.  It is therefore outside this actual-cost lane and is not searched.
LEDGER_DRIFT = {14: {"all_scores_cost": 356, "authority_profile_cost": 288}}

# Union of the maintained private-zero/unsound catalogue, the latest exact
# LB-black set, the six newly confirmed local-gold failures, and the expanded
# historical quarantine used by wave 408.
FAILED_GOLD = {12, 110, 161, 175, 188, 355}
PRIVATE_ZERO_OR_UNSOUND = set(BASE.PRIVATE_ZERO_OR_UNSOUND) | FAILED_GOLD
ELIGIBLE = tuple(task for task, _cost in BAND if task not in PRIVATE_ZERO_OR_UNSOUND)
COSTS = {task: cost for task, cost in BAND}

# Rebind every base-module global consumed by Worker methods.
BASE.HERE = HERE
BASE.AUTHORITY = AUTHORITY
BASE.AUTHORITY_SHA256 = AUTHORITY_SHA256
BASE.BAND = BAND
BASE.PRIVATE_ZERO_OR_UNSOUND = PRIVATE_ZERO_OR_UNSOUND
BASE.EXPLICIT_LATEST_LB_BLACK = {70, 134, 202, 343} | FAILED_GOLD
BASE.ELIGIBLE = ELIGIBLE
BASE.COSTS = COSTS
BASE.CHANGED_FROM_8011_05 = set()
BASE.THRESHOLD = 1.0
BASE.FRESH_PER_SEED = 2_000
BASE.SUPPORT.POLICY_THRESHOLD = 1.0
BASE.SUPPORT.FRESH_PER_SEED = 2_000

# ``evaluate_four`` is deliberately exhaustive and can spend minutes on a
# candidate that already fails the first fresh example.  At the exact-gold
# threshold, a one-runtime fail-fast rejection is conclusive.  Only candidates
# that survive every case proceed to the four-runtime determinism audit.
def exact_failfast_evaluate_four(
    data: bytes, cases: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    first = BASE.failfast_known(data, cases)
    # The official scorer and try_candidate gate use this sanitized raw ORT
    # configuration.  Returning immediately also makes exact failure at the
    # first fresh case genuinely fail-fast for large Einsum candidates.
    return {"disable_threads1": first}


BASE.SUPPORT.evaluate_four = exact_failfast_evaluate_four

# This archived task338 graph passes visible sign-gold but produced forbidden
# (0, 0.25) positives on the first case of each of two independent fresh-2000
# streams.  Quarantine it before the expensive full audit; the reproducible
# rejection is recorded in ``task338_fresh_reject.json``.
FRESH_REJECT_SHA256 = {
    "d7ea232e2e894d3749f4aebf67de754b29cba35f940147b9db314a59043781fe"
}
ORIGINAL_CONSIDER = BASE.Worker.consider


def consider_without_known_fresh_reject(
    self: Any, task: int, data: bytes, meta: dict[str, Any]
) -> None:
    key = hashlib.sha256(data).hexdigest()
    if key in FRESH_REJECT_SHA256:
        self.counters["known_fresh_sha_reject"] += 1
        self.seen[task].add(key)
        return
    ORIGINAL_CONSIDER(self, task, data, meta)


BASE.Worker.consider = consider_without_known_fresh_reject


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_gate(path: Path, task: int, authority_cost: int) -> dict[str, Any]:
    """Run the non-promoting equivalent of try_candidate.py's full gate."""
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
        gold_ok = False
        mismatch = None
        margin_ok = False
        min_positive = None
        score = None
        if structure_ok and model is not None:
            gold_ok, mismatch = TRY._verify_gold(model, task)
        if gold_ok and model is not None:
            margin_ok, min_positive = TRY._check_margin(model, task)
        if margin_ok and model is not None:
            with tempfile.TemporaryDirectory(prefix="gold351_500_") as workdir:
                score = TRY._score_model(
                    model, task, workdir, "candidate", require_correct=True
                )
    result.update(
        {
            "file_ok": file_ok,
            "structure_ok": structure_ok,
            "official_gold_exact": gold_ok,
            "margin_ok": margin_ok,
            "minimum_positive": min_positive,
            "scored": score is not None,
            "candidate_cost": None if score is None else int(score.cost),
            "strictly_cheaper": bool(score is not None and score.cost < authority_cost),
            "pass": bool(
                file_ok
                and structure_ok
                and gold_ok
                and margin_ok
                and score is not None
                and score.cost < authority_cost
            ),
            "first_mismatch": None
            if mismatch is None
            else {"subset": mismatch.subset, "index": mismatch.index},
            "try_candidate_log": stream.getvalue(),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    if digest(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("8012.23 authority SHA mismatch")

    payload = BASE.Worker(args.worker).run()
    gates: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for row in payload["finalists"]:
        path = ROOT / row["saved_path"]
        gate = official_gate(path, int(row["task"]), int(row["authority_cost"]))
        gates.append(gate)
        if gate["pass"]:
            row["official_try_candidate_gate"] = gate
            accepted.append(row)

    rejected_tasks = {int(gate["task"]) for gate in gates if not gate["pass"]}
    for task_row in payload["task_rows"]:
        if int(task_row["task"]) in rejected_tasks:
            task_row["admission"] = None
    payload["pre_official_gate_finalists"] = payload["finalists"]
    payload["official_gates"] = gates
    payload["finalists"] = accepted
    payload["threshold"] = 1.0
    payload["fresh_per_seed"] = 2_000
    payload["ledger_drift_excluded"] = LEDGER_DRIFT
    payload["absolute_admission_gate"] = (
        "try_candidate official gold exact + margin + structure + score; "
        "known and two fresh-2000 seeds all 100%"
    )
    payload["protected_writes"] = (
        "scripts/golf/cost351_500_gold_loop only; root submission/all_scores/"
        "best_score untouched"
    )

    output = HERE / f"worker_{args.worker}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "worker": args.worker,
                "tasks": payload["assigned_tasks"],
                "official_gold_finalists": [
                    {
                        "task": row["task"],
                        "cost": row["candidate_cost"],
                        "gain": row["score_gain"],
                    }
                    for row in accepted
                ],
                "official_gate_rejections": sorted(rejected_tasks),
                "elapsed": payload["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
