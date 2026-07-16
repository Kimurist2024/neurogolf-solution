#!/usr/bin/env python3
"""Rotate cost bands every 30 minutes and run evidence-only candidate scans.

This loop never promotes candidates or edits submission.zip/all_scores.csv. It
only creates per-cycle task manifests, scanner output, and image inventories
under others/71407/continuous_30m. A SHA guard stops the loop if root authority
files change unexpectedly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "others/71407/continuous_30m"
AUTHORITY = ROOT / "submission_base_8023.08.zip"
GUARDS = {
    # Current promoted candidate checkpoint (wave3); update deliberately so
    # the rotation can resume after an explicit submission refresh.
    "all_scores.csv": "81bfd10d12028f826c10eb41ce061936cee93e582750b07cff76ead44fb61b8b",
    "submission_base_8023.08.zip": "0e29e8d57f7ac58136a9574351c9c6f3056f9debf6eeee9c181c8f2e9fac690a",
}
KNOWN_BLACK = {
    9, 66, 70, 76, 93, 105, 134, 185, 187, 198, 199, 201, 202, 208,
    219, 233, 277, 286, 343, 365, 366, 396,
}
PRIVATE_ZERO = {
    9, 15, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    118, 133, 134, 138, 145, 157, 158, 168, 169, 170, 173, 174, 178, 182,
    185, 187, 191, 192, 196, 198, 202, 204, 205, 208, 209, 216, 219, 222,
    233, 246, 251, 255, 273, 277, 285, 286, 302, 319, 325, 333, 343, 346,
    361, 365, 366, 372, 377, 379, 391, 393, 396,
}
# Four-task focused rotation. task192 is explicitly whitelisted because its
# current cost-444 implementation is LB-confirmed white despite an old
# private-zero catalogue entry referring to a different historical model.
SEED_FOCUS_TASKS = (192, 243, 338, 344, 71, 175, 275, 354)
FOCUS_WHITE_OVERRIDE = {102, 168, 175, 192, 205, 219, 286, 354}
FAILED_GOLD_TASKS = {12, 110, 161, 188, 355}
FOCUS_STATE = OUT / "focus_registry.json"
AUTO_MIN_COST = 100
AUTO_MAX_COST = 500
AUTO_MAX_ACTIVE = 400
AUTO_ADD_PER_CYCLE = 5
STOP = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_guards() -> dict[str, str]:
    observed = {name: sha256(ROOT / name) for name in GUARDS}
    if observed != GUARDS:
        raise RuntimeError(f"root guard changed: {observed}")
    return observed


def current_costs() -> dict[int, int]:
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        return {
            int(row["task"].removeprefix("task")): int(row["cost"])
            for row in csv.DictReader(handle)
        }


def eligible_auto_pool(costs: dict[int, int]) -> list[int]:
    def allowed(task: int, cost: int) -> bool:
        if not AUTO_MIN_COST <= cost <= AUTO_MAX_COST:
            return False
        if (task in KNOWN_BLACK and task not in FOCUS_WHITE_OVERRIDE) or task in FAILED_GOLD_TASKS:
            return False
        if task in PRIVATE_ZERO and task not in FOCUS_WHITE_OVERRIDE:
            return False
        return True

    return [
        task for task, cost in sorted(costs.items(), key=lambda item: (-item[1], item[0]))
        if allowed(task, cost)
    ]


def write_focus_state(state: dict) -> None:
    FOCUS_STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = FOCUS_STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(FOCUS_STATE)


def load_focus_state(allow_add: bool = True) -> tuple[dict, dict[int, int]]:
    costs = current_costs()
    pool = eligible_auto_pool(costs)
    if FOCUS_STATE.is_file():
        try:
            state = json.loads(FOCUS_STATE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
    else:
        state = {}

    active = [
        int(task) for task in state.get("active_tasks", [])
        if int(task) in pool or int(task) in FOCUS_WHITE_OVERRIDE
    ]
    for task in SEED_FOCUS_TASKS:
        if task not in active:
            active.append(task)
    active = list(dict.fromkeys(active))
    retired = [int(task) for task in state.get("retired_tasks", [])]
    visits = {str(task): int(count) for task, count in state.get("visits", {}).items()}
    no_gold = {
        str(task): int(count) for task, count in state.get("no_gold_streak", {}).items()
    }
    additions = list(state.get("auto_additions", []))

    available = [task for task in pool if task not in active and task not in retired]
    added_now = []
    while (
        allow_add
        and available
        and len(active) < AUTO_MAX_ACTIVE
        and len(added_now) < AUTO_ADD_PER_CYCLE
    ):
        task = available.pop(0)
        active.append(task)
        added_now.append(task)
        additions.append({
            "task": task,
            "cost": costs[task],
            "utc": datetime.now(timezone.utc).isoformat(),
            "reason": "automatic_high_cost_unexplored_replenishment",
        })

    state = {
        "policy": {
            "cost_range": [AUTO_MIN_COST, AUTO_MAX_COST],
            "max_active": AUTO_MAX_ACTIVE,
            "add_per_cycle": AUTO_ADD_PER_CYCLE,
            "gold_required": True,
        },
        "active_tasks": active,
        "retired_tasks": retired,
        "queued_tasks": available,
        "visits": visits,
        "no_gold_streak": no_gold,
        "auto_additions": additions,
        "added_this_refresh": added_now,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_focus_state(state)
    return state, costs


def select_focus_task() -> int:
    state, costs = load_focus_state()
    active = [int(task) for task in state["active_tasks"]]
    if not active:
        raise RuntimeError("autonomous focus pool is empty")
    visits = state["visits"]
    selected = min(
        active,
        key=lambda task: (int(visits.get(str(task), 0)), -int(costs.get(task, 0)), task),
    )
    state["selected_task"] = selected
    state["selected_utc"] = datetime.now(timezone.utc).isoformat()
    write_focus_state(state)
    return selected


def record_focus_result(task: int, gold_finalist_count: int) -> None:
    state, _ = load_focus_state(allow_add=False)
    key = str(task)
    state["visits"][key] = int(state["visits"].get(key, 0)) + 1
    if gold_finalist_count:
        state["no_gold_streak"][key] = 0
    else:
        state["no_gold_streak"][key] = int(state["no_gold_streak"].get(key, 0)) + 1
    # Seed tasks remain available for deep specialized work. Automatically
    # added tasks rotate out after two fruitless focused passes.
    if (
        task not in SEED_FOCUS_TASKS
        and state["no_gold_streak"][key] >= 2
        and task in state["active_tasks"]
    ):
        state["active_tasks"].remove(task)
        if task not in state["retired_tasks"]:
            state["retired_tasks"].append(task)
    state["last_result"] = {
        "task": task,
        "gold_finalist_count": gold_finalist_count,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    write_focus_state(state)


def task_rows(selected_task: int) -> list[dict]:
    rows = []
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"].removeprefix("task"))
            cost = int(row["cost"])
            if task == selected_task:
                row["task_id"] = task
                row["excluded"] = (task in KNOWN_BLACK and task not in FOCUS_WHITE_OVERRIDE) or (
                    task in PRIVATE_ZERO and task not in FOCUS_WHITE_OVERRIDE
                )
                row["image"] = str(ROOT / f"artifacts/task_viz/task{task:03d}.png")
                row["image_exists"] = Path(row["image"]).is_file()
                rows.append(row)
    return rows


def collect_gold_finalists(cycle_dir: Path) -> list[dict]:
    finalists: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for path in sorted((cycle_dir / "repair_3w").glob("worker_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("finalists", []):
            task = int(row.get("task", 0))
            sha = str(row.get("sha256", ""))
            key = (task, sha)
            if not task or not sha or key in seen:
                continue
            if row.get("official_gold") is not True:
                continue
            fresh = row.get("fresh", [])
            if len(fresh) != 2 or not all(item.get("pass") for item in fresh):
                continue
            seen.add(key)
            finalists.append(row)
    return finalists


def run_cycle(cycle: int, selected_task: int, timeout: int) -> dict:
    check_guards()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cycle_dir = OUT / f"cycle_{cycle:04d}_{stamp}_task{selected_task:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=False)
    rows = task_rows(selected_task)
    eligible = [row for row in rows if not row["excluded"]]
    manifest = {
        "cycle": cycle,
        "started_utc": stamp,
        "focus_task": selected_task,
        "authority": {"path": str(AUTHORITY.relative_to(ROOT)), "sha256": GUARDS["submission_base_8023.08.zip"]},
        "known_black_excluded": sorted(KNOWN_BLACK),
        "private_zero_excluded": sorted(PRIVATE_ZERO),
        "task_count": len(rows),
        "eligible_count": len(eligible),
        "tasks": rows,
        "image_inference": {
            "directory": "artifacts/task_viz",
            "available_count": sum(bool(row["image_exists"]) for row in rows),
            "rule": "visual template classification is evidence-only until runtime gates pass",
        },
    }
    (cycle_dir / "TASK_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output = cycle_dir / "scan_relaxed95_candidates.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/golf/scan_relaxed95_candidates.py"),
        "--low", "1",
        "--high", "1000000000",
        "--tasks", str(selected_task),
        "--timeout",
        str(timeout),
        "--output",
        str(output),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout * max(1, len(eligible)) + 120,
            check=False,
        )
        status = completed.returncode
        (cycle_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (cycle_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        status = 124
        (cycle_dir / "stdout.log").write_text(exc.stdout or "", encoding="utf-8")
        (cycle_dir / "stderr.log").write_text(str(exc), encoding="utf-8")
    feedback = build_feedback(output, rows, cycle_dir, selected_task, timeout)
    gold_finalists = collect_gold_finalists(cycle_dir)
    auto_stage = None
    if gold_finalists:
        try:
            from autostage_verified_candidates import stage_candidates

            stage_rows = [
                {
                    "task": int(row["task"]),
                    "path": row["saved_path"],
                    "sha256": row["sha256"],
                    "authority_cost": int(row["authority_cost"]),
                    "candidate_cost": int(row["candidate_cost"]),
                    "strict_gate": True,
                    "evidence": str(cycle_dir.relative_to(ROOT)),
                }
                for row in gold_finalists
            ]
            staged = stage_candidates(stage_rows)
            auto_stage = {
                "pass": True,
                "projected_gain": staged["projected_gain"],
                "projected_score": staged["projected_score"],
                "submission_sha256": staged["submission"]["sha256"],
            }
        except Exception as exc:  # admission failure is retained as evidence
            auto_stage = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
    check_guards()
    result = {
        "cycle": cycle,
        "focus_task": selected_task,
        "tasks": len(rows),
        "eligible": len(eligible),
        "returncode": status,
        "elapsed_seconds": time.monotonic() - started,
        "output": str(output.relative_to(ROOT)),
        "feedback": feedback,
        "gold_finalists": gold_finalists,
        "auto_stage": auto_stage,
        "root_modified": bool(auto_stage and auto_stage.get("pass")),
    }
    (cycle_dir / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (OUT / "rotation_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    record_focus_result(selected_task, len(gold_finalists))
    return result


def build_feedback(
    output: Path,
    rows: list[dict],
    cycle_dir: Path,
    selected_task: int,
    timeout: int,
) -> dict:
    """Turn a failed scan into an explicit next-pass repair plan.

    The first pass may have no history files in a cost band.  In that case a
    bounded rescan of recent restart/checkpoint lanes is performed.  All
    failures are retained as data for the next 30-minute rotation.
    """
    payload: dict = {}
    if output.is_file():
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    scan_rows = payload.get("rows", [])
    reasons = {
        "no_candidates": int(payload.get("candidate_count", 0)) == 0,
        "runtime_or_load_error": sum("error" in row for row in scan_rows),
        "visible_incorrect": sum(not row.get("correct", False) for row in scan_rows),
        "no_strict_cost_gain": sum(
            row.get("correct") and "gain" not in row for row in scan_rows
        ),
    }
    actions: list[str] = []
    if reasons["no_candidates"]:
        actions.append("expand_recent_history_sources")
    if reasons["runtime_or_load_error"]:
        actions.append("quarantine_runtime_and_shape_failures")
    if reasons["visible_incorrect"]:
        actions.append("prioritize_image_rule_review_and_rebuild")
    if reasons["no_strict_cost_gain"]:
        actions.append("try_structural_shave_before_new_training")

    repair_queue = {
        "focus_task": selected_task,
        "actions": actions,
        "tasks": [
            {
                "task": row["task_id"],
                "cost": int(row["cost"]),
                "image": row["image"],
                "image_exists": row["image_exists"],
                "reason": ";".join(actions) or "none",
            }
            for row in rows
            if not row["excluded"]
        ],
        "policy": "official gold exact match required; runtime/nonfinite/shape/margin errors are hard reject",
    }
    (cycle_dir / "REPAIR_QUEUE.json").write_text(
        json.dumps(repair_queue, indent=2) + "\n", encoding="utf-8"
    )

    feedback_scan = None
    if reasons["no_candidates"] or reasons["runtime_or_load_error"]:
        sources = [
            ROOT / "scripts/golf/restart8012_pending_3w_404",
            ROOT / "scripts/golf/restart8012_task354_main_407",
            ROOT / "scripts/golf/restart8012_dedup_main_412",
            ROOT / "scripts/golf/restart8012_scalar_dedup_main_413",
            ROOT / "others/71407/nonblack_policy90_8012_15_wave2",
        ]
        sources = [path for path in sources if path.exists()]
        feedback_output = cycle_dir / "feedback_rescan.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/golf/scan_relaxed95_candidates.py"),
            *[str(path.relative_to(ROOT)) for path in sources],
            "--low", "1",
            "--high", "1000000000",
            "--tasks", str(selected_task),
            "--timeout",
            str(min(timeout, 5)),
            "--rescan",
            "--output",
            str(feedback_output),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            (cycle_dir / "feedback_stdout.log").write_text(completed.stdout, encoding="utf-8")
            (cycle_dir / "feedback_stderr.log").write_text(completed.stderr, encoding="utf-8")
            feedback_scan = {
                "returncode": completed.returncode,
                "output": str(feedback_output.relative_to(ROOT)),
                "sources": [str(path.relative_to(ROOT)) for path in sources],
            }
        except subprocess.TimeoutExpired as exc:
            (cycle_dir / "feedback_stdout.log").write_text(exc.stdout or "", encoding="utf-8")
            (cycle_dir / "feedback_stderr.log").write_text(str(exc), encoding="utf-8")
            feedback_scan = {"returncode": 124, "sources": [str(path.relative_to(ROOT)) for path in sources]}

    repair_workers = None
    if reasons["no_candidates"] or reasons["runtime_or_load_error"] or reasons["visible_incorrect"]:
        repair_dir = cycle_dir / "repair_3w"
        repair_dir.mkdir(parents=True, exist_ok=True)
        processes = []
        command_base = [
            sys.executable,
            str(ROOT / "scripts/golf/repair_feedback_3w.py"),
            "--low", "1",
            "--high", "1000000000",
            "--tasks", str(selected_task),
            "--output",
            str(repair_dir),
        ]
        for worker_id in range(3):
            log = (repair_dir / f"worker_{worker_id}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                [*command_base, "--worker", str(worker_id)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((worker_id, process, log))
        statuses = []
        deadline = time.monotonic() + 900
        for worker_id, process, log in processes:
            remaining = max(1, deadline - time.monotonic())
            try:
                status = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.terminate()
                status = 124
            log.close()
            statuses.append({"worker": worker_id, "returncode": status})
        repair_workers = {
            "directory": str(repair_dir.relative_to(ROOT)),
            "workers": statuses,
            "policy": "official gold (train/test/arc-gen) required; fresh/runtime gates are secondary",
        }

    feedback = {
        "reasons": reasons,
        "actions": actions,
        "feedback_rescan": feedback_scan,
        "repair_workers": repair_workers,
    }
    (cycle_dir / "FEEDBACK.json").write_text(json.dumps(feedback, indent=2) + "\n", encoding="utf-8")
    return feedback


def stop_handler(_signum, _frame) -> None:
    global STOP
    STOP = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--start-cycle", type=int, default=0)
    args = parser.parse_args()
    if args.interval < 60:
        raise SystemExit("--interval must be at least 60 seconds")
    OUT.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    cycle = args.start_cycle
    while not STOP:
        cycle_started = time.monotonic()
        selected_task = select_focus_task()
        result = run_cycle(cycle, selected_task, args.timeout)
        print(json.dumps(result), flush=True)
        cycle += 1
        if args.once or STOP:
            break
        deadline = cycle_started + args.interval
        while not STOP and time.monotonic() < deadline:
            time.sleep(min(30, max(1, deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
