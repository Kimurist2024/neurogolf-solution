#!/usr/bin/env python3
"""Small locked state helper for the detached factory daemon."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
FACTORY_DIR = REPO_ROOT / "artifacts" / "factory"
STATE_PATH = FACTORY_DIR / "state.json"
LOCK_PATH = FACTORY_DIR / "state.lock"
QUEUE_PATH = REPO_ROOT / "docs" / "golf" / "queue.json"
STATUS_PATH = FACTORY_DIR / "status.json"
PAUSE_PATH = FACTORY_DIR / "PAUSE"
MERGE_REPORT = REPO_ROOT / "artifacts" / "reports" / "merge-001.json"
BEST_SCORE = REPO_ROOT / "artifacts" / "best_score.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


@contextmanager
def _locked() -> Iterator[None]:
    FACTORY_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _empty_state() -> dict[str, Any]:
    return {
        "created_at": _now(),
        "updated_at": _now(),
        "queue_index": 0,
        "in_progress": [],
        "done": [],
        "done_details": {},
        "history": [],
        "promotions": 0,
        "retries": {},
    }


def _state() -> dict[str, Any]:
    state = _read_json(STATE_PATH, _empty_state())
    for key, value in _empty_state().items():
        state.setdefault(key, value)
    state["in_progress"] = list(state.get("in_progress", []))
    state["done"] = [int(task) for task in state.get("done", [])]
    state["history"] = list(state.get("history", []))
    state["done_details"] = dict(state.get("done_details", {}))
    state["promotions"] = int(state.get("promotions", 0))
    state["retries"] = dict(state.get("retries", {}))
    return state


def _queue() -> list[dict[str, Any]]:
    return list(_read_json(QUEUE_PATH, []))


def _pid_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except OSError:
        return False
    return True


def _release_stale(state: dict[str, Any]) -> None:
    active: list[dict[str, Any]] = []
    for item in state["in_progress"]:
        pid = item.get("pid")
        if pid is None or _pid_alive(pid):
            active.append(item)
            continue
        stale = dict(item)
        stale["event"] = "stale_released"
        stale["released_at"] = _now()
        state["history"].append(stale)
    state["in_progress"] = active


def _promotion_count(state: dict[str, Any]) -> int:
    return sum(1 for item in state.get("history", [])
               if bool(item.get("promoted")))


def _save_state(state: dict[str, Any]) -> None:
    state["promotions"] = _promotion_count(state)
    state["updated_at"] = _now()
    _write_json(STATE_PATH, state)


def _maybe_pause_after_failures(state: dict[str, Any]) -> None:
    # Only genuine worker finishes count; stale_released/released events have
    # no exit_code and previously masked consecutive failures (56 burned
    # before the brake engaged in the 2026-06-13 incident).
    finishes = [item for item in state.get("history", [])
                if "finished_at" in item]
    recent = finishes[-8:]
    if len(recent) < 8:
        return
    if all(int(item.get("exit_code", 0)) != 0 and not item.get("promoted")
           for item in recent):
        PAUSE_PATH.write_text(
            "Paused automatically after 8 consecutive worker failures. "
            "Fix the worker environment, then run "
            ".venv/bin/python scripts/factory/state.py resume.\n",
            encoding="utf-8",
        )


def cmd_init(_args: argparse.Namespace) -> int:
    with _locked():
        state = _state()
        _release_stale(state)
        _save_state(state)
    return 0


def cmd_claim(args: argparse.Namespace) -> int:
    with _locked():
        if PAUSE_PATH.exists():
            return 2
        queue = _queue()
        state = _state()
        _release_stale(state)
        if not queue:
            _save_state(state)
            return 2

        done = set(int(task) for task in state["done"])
        active = set(int(item["task"]) for item in state["in_progress"])
        start = int(state.get("queue_index", 0)) % len(queue)
        chosen: dict[str, Any] | None = None
        chosen_idx = start
        for offset in range(len(queue)):
            idx = (start + offset) % len(queue)
            item = queue[idx]
            task = int(item["task"])
            if task in done or task in active:
                continue
            chosen = {
                "task": task,
                "hash": str(item["hash"]),
                "cost": int(item["cost"]),
                "slot": args.slot,
                "pid": None,
                "claimed_at": _now(),
            }
            chosen_idx = idx
            break

        if chosen is None:
            _save_state(state)
            return 2

        state["in_progress"].append(chosen)
        state["queue_index"] = (chosen_idx + 1) % len(queue)
        _save_state(state)

    if args.tsv:
        print(f"{chosen['task']} {chosen['hash']} {chosen['cost']}")
    else:
        print(json.dumps(chosen, sort_keys=True))
    return 0


def cmd_set_pid(args: argparse.Namespace) -> int:
    with _locked():
        state = _state()
        task = int(args.task)
        for item in state["in_progress"]:
            if int(item["task"]) == task:
                item["pid"] = int(args.pid)
                item["started_at"] = _now()
                break
        _save_state(state)
    return 0


def cmd_finish(args: argparse.Namespace) -> int:
    with _locked():
        state = _state()
        task = int(args.task)
        remaining = []
        finished: dict[str, Any] = {}
        for item in state["in_progress"]:
            if int(item["task"]) == task:
                finished = dict(item)
            else:
                remaining.append(item)
        state["in_progress"] = remaining

        promoted = args.promoted.lower() == "true"
        exit_code = int(args.exit_code)
        key = f"{task:03d}"
        retry_count = int(state["retries"].get(key, 0))
        if exit_code != 0 and not promoted and retry_count < 2:
            # Failed run: leave the task claimable instead of burning it.
            state["retries"][key] = retry_count + 1
        elif task not in state["done"]:
            state["done"].append(task)
            state["done"].sort()
        result = {
            **finished,
            "task": task,
            "exit_code": exit_code,
            "promoted": promoted,
            "new_cost": None if args.cost is None else int(args.cost),
            "finished_at": _now(),
        }
        state["done_details"][f"{task:03d}"] = result
        state["history"].append(result)
        _maybe_pause_after_failures(state)
        _save_state(state)
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    """Drop a claim without consuming the task (preflight failures etc.)."""
    with _locked():
        state = _state()
        task = int(args.task)
        remaining = []
        released: dict[str, Any] = {}
        for item in state["in_progress"]:
            if int(item["task"]) == task:
                released = dict(item)
            else:
                remaining.append(item)
        state["in_progress"] = remaining
        if released:
            released["event"] = "released"
            released["released_at"] = _now()
            state["history"].append(released)
        _save_state(state)
    return 0


def cmd_promotions(_args: argparse.Namespace) -> int:
    with _locked():
        state = _state()
        print(_promotion_count(state))
    return 0


def cmd_reset_runtime(_args: argparse.Namespace) -> int:
    with _locked():
        state = _empty_state()
        _save_state(state)
    return 0


def cmd_pause(args: argparse.Namespace) -> int:
    PAUSE_PATH.write_text(args.reason + "\n", encoding="utf-8")
    return 0


def cmd_resume(_args: argparse.Namespace) -> int:
    try:
        PAUSE_PATH.unlink()
    except FileNotFoundError:
        pass
    return 0


def _epoch_to_iso(value: int) -> str | None:
    if value <= 0:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(
        timespec="seconds"
    )


def _expected_score() -> dict[str, Any]:
    if MERGE_REPORT.is_file():
        report = _read_json(MERGE_REPORT, {})
        totals = report.get("totals", {})
        if "score_after" in totals:
            return {
                "source": str(MERGE_REPORT.relative_to(REPO_ROOT)),
                "score": totals.get("score_after"),
                "delta": totals.get("score_delta"),
            }
    best = _read_json(BEST_SCORE, {})
    return {
        "source": str(BEST_SCORE.relative_to(REPO_ROOT)),
        "score": best.get("score"),
        "delta": None,
    }


def cmd_status_json(args: argparse.Namespace) -> int:
    with _locked():
        queue = _queue()
        state = _state()
        _release_stale(state)
        done = set(int(task) for task in state["done"])
        active_tasks = set(int(item["task"]) for item in state["in_progress"])
        remaining = sum(
            1 for item in queue
            if int(item["task"]) not in done and int(item["task"]) not in active_tasks
        )
        promotions = _promotion_count(state)
        status = {
            "generated_at": _now(),
            "driver_pid": int(args.driver_pid),
            "paused": PAUSE_PATH.exists(),
            "workers_configured": int(args.workers),
            "active_workers": state["in_progress"],
            "queue": {
                "total": len(queue),
                "remaining": remaining,
                "index": int(state.get("queue_index", 0)),
            },
            "done_count": len(done),
            "promotions": promotions,
            "promotions_since_last_merge": max(
                0, promotions - int(args.promotions_at_last_merge)
            ),
            "last_merge": {
                "at": _epoch_to_iso(int(args.last_merge_epoch)),
                "status": args.last_merge_status,
            },
            "last_submit": {
                "at": _epoch_to_iso(int(args.last_submit_epoch)),
                "status": args.last_submit_status,
            },
            "expected_score": _expected_score(),
            "recent_results": state["history"][-8:],
        }
        _write_json(STATUS_PATH, status)
        _save_state(state)
    print(json.dumps(status, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init")
    init.set_defaults(func=cmd_init)

    claim = sub.add_parser("claim")
    claim.add_argument("--slot", type=int, required=True)
    claim.add_argument("--tsv", action="store_true")
    claim.set_defaults(func=cmd_claim)

    set_pid = sub.add_parser("set-pid")
    set_pid.add_argument("--task", type=int, required=True)
    set_pid.add_argument("--pid", type=int, required=True)
    set_pid.set_defaults(func=cmd_set_pid)

    finish = sub.add_parser("finish")
    finish.add_argument("--task", type=int, required=True)
    finish.add_argument("--exit-code", type=int, required=True)
    finish.add_argument("--promoted", choices=["true", "false"], required=True)
    finish.add_argument("--cost", type=int)
    finish.set_defaults(func=cmd_finish)

    release = sub.add_parser("release")
    release.add_argument("--task", type=int, required=True)
    release.set_defaults(func=cmd_release)

    promotions = sub.add_parser("promotions")
    promotions.set_defaults(func=cmd_promotions)

    reset = sub.add_parser("reset-runtime")
    reset.set_defaults(func=cmd_reset_runtime)

    pause = sub.add_parser("pause")
    pause.add_argument("reason", nargs="?", default="Paused manually.")
    pause.set_defaults(func=cmd_pause)

    resume = sub.add_parser("resume")
    resume.set_defaults(func=cmd_resume)

    status = sub.add_parser("status-json")
    status.add_argument("--workers", type=int, required=True)
    status.add_argument("--driver-pid", type=int, required=True)
    status.add_argument("--last-merge-epoch", type=int, required=True)
    status.add_argument("--last-submit-epoch", type=int, required=True)
    status.add_argument("--last-merge-status", default="never")
    status.add_argument("--last-submit-status", default="never")
    status.add_argument("--promotions-at-last-merge", type=int, required=True)
    status.set_defaults(func=cmd_status_json)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
