#!/usr/bin/env python3
"""8019.75 evidence-only high-cost/new-drop search.

Scans every new 71604/71605 loose candidate and bundle member for authority
tasks at cost >=300, then runs exact current-authority simplifiers.  Admission
is local+official gold exact, strict/static, stable margin, and two independent
fresh-2000 streams at 100%.  task275 is excluded because it is already staged.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import zipfile
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
    "restart8019_75_high_base",
    ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py",
)
TRY = import_path(
    "restart8019_75_high_try",
    ROOT / "scripts/golf/try_candidate.py",
)
VERIFY = import_path(
    "restart8019_75_high_verify",
    ROOT / "scripts/verify_fix.py",
)

AUTHORITY = ROOT / "submission_base_8019.75.zip"
AUTHORITY_SHA256 = "e69058edd21e27ab7d32670d714ec5cea6d35632a9d9a620364731297717edb3"
NEW_DIRS = (ROOT / "others/71604", ROOT / "others/71605")


def current_band() -> tuple[tuple[int, int], ...]:
    rows: list[tuple[int, int]] = []
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"][4:])
            cost = int(row["cost"])
            if cost >= 300 and task != 275:
                rows.append((task, cost))
    return tuple(sorted(rows, key=lambda item: (-item[1], item[0])))


BAND = current_band()
ELIGIBLE = tuple(task for task, _cost in BAND)
COSTS = dict(BAND)

BASE.HERE = HERE
BASE.AUTHORITY = AUTHORITY
BASE.AUTHORITY_SHA256 = AUTHORITY_SHA256
BASE.BAND = BAND
BASE.PRIVATE_ZERO_OR_UNSOUND = set()
BASE.EXPLICIT_LATEST_LB_BLACK = set()
BASE.ELIGIBLE = ELIGIBLE
BASE.COSTS = COSTS
BASE.CHANGED_FROM_8011_05 = set(ELIGIBLE)
BASE.THRESHOLD = 1.0
BASE.FRESH_PER_SEED = 2_000
BASE.SUPPORT.POLICY_THRESHOLD = 1.0
BASE.SUPPORT.FRESH_PER_SEED = 2_000


def exact_failfast_evaluate(
    data: bytes, cases: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {"disable_threads1": BASE.failfast_known(data, cases)}


BASE.SUPPORT.evaluate_four = exact_failfast_evaluate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_gate(path: Path, task: int, authority_cost: int) -> dict[str, Any]:
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
        official_gold = False
        margin_ok = False
        minimum_positive = None
        score = None
        if structure_ok and model is not None:
            local_gold, mismatch = TRY._verify_gold(model, task)
        if local_gold:
            official_gold = VERIFY.official_gold(path, task)
        if official_gold and model is not None:
            margin_ok, minimum_positive = TRY._check_margin(model, task)
        if margin_ok and model is not None:
            with tempfile.TemporaryDirectory(prefix="high8019_gate_") as workdir:
                score = TRY._score_model(
                    model, task, workdir, "candidate", require_correct=True
                )
    passed = bool(
        file_ok and structure_ok and local_gold and official_gold and margin_ok
        and score is not None and score.cost < authority_cost
    )
    result.update({
        "file_ok": file_ok,
        "structure_ok": structure_ok,
        "local_gold_exact": local_gold,
        "official_gold_exact": official_gold,
        "margin_ok": margin_ok,
        "minimum_positive": minimum_positive,
        "candidate_cost": None if score is None else int(score.cost),
        "strictly_cheaper": bool(score is not None and score.cost < authority_cost),
        "pass": passed,
        "first_mismatch": None if mismatch is None else {
            "subset": mismatch.subset, "index": mismatch.index
        },
        "gate_log": stream.getvalue(),
    })
    return result


def task_from_filename(name: str) -> int | None:
    match = re.search(r"task[_-]?(\d{3})(?!\d)", Path(name).name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def scan_new_drops(worker: Any) -> None:
    for directory in NEW_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.onnx")):
            task = task_from_filename(path.name)
            if task not in worker.task_set:
                continue
            worker.consider(task, path.read_bytes(), {
                "name": path.name,
                "family": "new_drop_71604_71605",
                "detail": "new loose candidate after 8018.91 sweep",
                "source": str(path.relative_to(ROOT)),
            })
        for path in sorted(directory.glob("*.zip")):
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        task = task_from_filename(member)
                        if task not in worker.task_set or not member.endswith(".onnx"):
                            continue
                        worker.consider(task, archive.read(member), {
                            "name": Path(member).name,
                            "family": "new_bundle_71604_71605",
                            "detail": "new bundle member after 8018.91 sweep",
                            "source": f"{path.relative_to(ROOT)}!{member}",
                        })
            except Exception:
                worker.counters["new_zip_error"] += 1


def reprofile_authority(worker: Any) -> None:
    """Bind comparisons to measured authority cost, never the CSV estimate."""
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in worker.tasks:
            data = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_model_from_string(data)
            cases, _counts = worker.cases[task]
            profile = BASE.POLICY.fast_profile(BASE.SUPPORT, task, model, cases[0])
            if profile is None:
                raise RuntimeError(f"task{task:03d}: authority did not profile")
            measured = int(profile["cost"])
            ledger = int(worker.task_rows[task]["authority_cost"])
            COSTS[task] = measured
            worker.task_rows[task]["authority_cost"] = measured
            worker.task_rows[task]["authority_reprofile"] = {
                "sha256": BASE.digest(data),
                "profile": profile,
                "ledger_cost": ledger,
                "measured_cost": measured,
                "ledger_matches": ledger == measured,
            }
            worker.seen[task].add(BASE.digest(data))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if digest(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("8019.75 authority SHA mismatch")

    worker = BASE.Worker(args.worker)
    reprofile_authority(worker)
    scan_new_drops(worker)
    worker.scan_current_simplifiers()
    pre = worker.full_audit()
    gates = []
    accepted = []
    for row in pre:
        path = ROOT / row["saved_path"]
        gate = strict_gate(path, int(row["task"]), int(row["authority_cost"]))
        gates.append(gate)
        if gate["pass"]:
            row["strict_gate"] = gate
            accepted.append(row)
    payload = {
        "worker": args.worker,
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "assigned_tasks": list(worker.tasks),
        "band_count": len(BAND),
        "task_rows": [worker.task_rows[task] for task in worker.tasks],
        "counters": dict(worker.counters),
        "pre_strict_finalists": pre,
        "strict_gates": gates,
        "finalists": accepted,
        "absolute_gate": (
            "local+official gold exact, strict/static, stable margin, "
            "fresh2000x2 100%, zero runtime/nonfinite/shape/small-positive"
        ),
        "protected_writes": "lane only; root authority unchanged",
    }
    (HERE / f"worker_{args.worker}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "worker": args.worker,
        "tasks": len(worker.tasks),
        "strict_winners": [
            {"task": row["task"], "cost": row["candidate_cost"],
             "gain": row["score_gain"], "path": row["saved_path"]}
            for row in accepted
        ],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
