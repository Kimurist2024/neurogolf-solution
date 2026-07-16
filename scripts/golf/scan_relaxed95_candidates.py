#!/usr/bin/env python3
"""Find cheaper, visible-correct ONNX candidates without merging them.

This is the discovery stage of the user-authorized 95% improvement loop.  It
never edits ``submission.zip`` or ``artifacts/handcrafted``.  Each candidate is
scored in an isolated subprocess to avoid ONNX Runtime state contamination.
The resulting shortlist is intended for ``scripts/verify_fix.py --k 5000
--min-fresh-rate 0.95``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import tempfile
from pathlib import Path

import onnx


REPO = Path(__file__).resolve().parents[2]
TASK_RE = re.compile(r"task[_-]?(\d{1,3})", re.IGNORECASE)
DEFAULT_SOURCES = ("others/1200", "others/1201", "others/1202", "others/1102")


def load_costs(path: Path, incumbents_path: Path) -> dict[int, int]:
    with path.open(newline="") as handle:
        costs = {
            int(row["task"].removeprefix("task")): int(row["cost"])
            for row in csv.DictReader(handle)
        }
    if incumbents_path.exists():
        payload = json.loads(incumbents_path.read_text())
        for task, cost in payload.get("costs", {}).items():
            task_number = int(task)
            costs[task_number] = min(costs.get(task_number, int(cost)), int(cost))
    return costs


def load_prior_hashes(results_dir: Path) -> set[tuple[int, str]]:
    """Recover evaluated candidate hashes from earlier discovery rounds."""
    seen: set[tuple[int, str]] = set()
    for result_path in sorted(results_dir.glob("*.json")):
        try:
            rows = json.loads(result_path.read_text()).get("rows", [])
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            path = REPO / row.get("path", "")
            task = row.get("task")
            if not task or not path.is_file():
                continue
            seen.add((int(task), hashlib.sha1(path.read_bytes()).hexdigest()))
    return seen


def worker(path: str, task: int, queue: mp.Queue) -> None:
    try:
        # Broken historical candidates can make ONNX Runtime emit thousands of
        # native stderr lines.  Candidate failures are recorded via the queue,
        # so silence the isolated child before importing/running the runtime.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)

        import onnxruntime

        onnxruntime.set_default_logger_severity(3)
        sys.path.insert(0, str(REPO / "scripts"))
        from lib import scoring

        with tempfile.TemporaryDirectory() as workdir:
            result = scoring.score_and_verify(
                onnx.load(path),
                task,
                workdir,
                label="relaxed95-discovery",
                require_correct=True,
            )
        queue.put(
            {
                "cost": result["cost"] if result else None,
                "correct": bool(result and result.get("correct")),
            }
        )
    except Exception as exc:  # noqa: BLE001 - candidate failures are data
        queue.put({"cost": None, "correct": False, "error": type(exc).__name__})


def score_isolated(path: Path, task: int, timeout: int) -> dict:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=worker, args=(str(path), task, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(3)
        if process.is_alive():
            process.kill()
        return {"cost": None, "correct": False, "error": "timeout"}
    try:
        return queue.get_nowait()
    except Exception:  # noqa: BLE001 - a crashed candidate may not fill queue
        return {"cost": None, "correct": False, "error": "no_result"}


def discover(
    sources: list[Path],
    costs: dict[int, int],
    low: int,
    high: int,
    seen: set[tuple[int, str]],
    selected_tasks: set[int] | None = None,
):
    candidates: list[tuple[int, Path]] = []
    for source in sources:
        if not source.exists():
            continue
        paths = [source] if source.is_file() and source.suffix == ".onnx" else source.rglob("*.onnx")
        for path in sorted(paths):
            match = TASK_RE.search(path.name)
            if not match:
                continue
            task = int(match.group(1))
            if selected_tasks is not None and task not in selected_tasks:
                continue
            base_cost = costs.get(task)
            if base_cost is None or not low <= base_cost <= high:
                continue
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
            key = (task, digest)
            if key not in seen:
                seen.add(key)
                candidates.append((task, path))
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="*", default=list(DEFAULT_SOURCES))
    parser.add_argument("--low", type=int, default=150)
    parser.add_argument("--high", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--tasks", type=int, nargs="*", default=None)
    parser.add_argument(
        "--incumbents",
        type=Path,
        default=REPO / "artifacts" / "relaxed95_loop" / "incumbents.json",
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="ignore hashes recorded by earlier relaxed95 discovery rounds",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "artifacts" / "relaxed95_loop" / "discovery.json",
    )
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = REPO / args.output
    if not args.incumbents.is_absolute():
        args.incumbents = REPO / args.incumbents

    costs = load_costs(REPO / "all_scores.csv", args.incumbents)
    sources = [REPO / source for source in args.sources]
    seen = set() if args.rescan else load_prior_hashes(args.output.parent)
    prior_count = len(seen)
    selected_tasks = set(args.tasks) if args.tasks else None
    candidates = discover(sources, costs, args.low, args.high, seen, selected_tasks)
    print(f"prior_hashes={prior_count}", flush=True)
    print(f"candidates={len(candidates)}", flush=True)

    rows: list[dict] = []
    for index, (task, path) in enumerate(candidates, 1):
        result = score_isolated(path, task, args.timeout)
        base_cost = costs[task]
        candidate_cost = result.get("cost")
        row = {
            "task": task,
            "base_cost": base_cost,
            "candidate_cost": candidate_cost,
            "correct": result.get("correct", False),
            "path": str(path.relative_to(REPO)),
        }
        if result.get("error"):
            row["error"] = result["error"]
        if result.get("correct") and isinstance(candidate_cost, int) and candidate_cost < base_cost:
            row["gain"] = math.log(base_cost / candidate_cost)
            print(
                f"WIN task{task:03d} {base_cost}->{candidate_cost} "
                f"+{row['gain']:.6f} {row['path']}",
                flush=True,
            )
        rows.append(row)
        if index % 20 == 0:
            print(f"progress={index}/{len(candidates)}", flush=True)

    winners = [row for row in rows if "gain" in row]
    winners.sort(key=lambda row: (-row["gain"], row["task"]))
    payload = {
        "policy": {"fresh_rate": 0.95, "fresh_k": 5000},
        "range": [args.low, args.high],
        "tasks": sorted(selected_tasks) if selected_tasks is not None else None,
        "sources": [str(path.relative_to(REPO)) for path in sources],
        "candidate_count": len(candidates),
        "winners": winners,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"winners={len(winners)} visible_gain=+"
        f"{sum(row['gain'] for row in winners):.6f}",
        flush=True,
    )
    print(f"output={args.output.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
