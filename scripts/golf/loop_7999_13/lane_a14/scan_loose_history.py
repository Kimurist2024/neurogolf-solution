#!/usr/bin/env python3
"""Deduplicated loose ONNX history scan for A14 tasks."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
from harvest import actual_screen, structure_gate  # noqa: E402
from lib import scoring  # noqa: E402


TASKS = (119, 153, 161, 174, 183, 190, 243)
BASE_COST = {119: 142, 153: 237, 161: 190, 174: 240, 183: 162, 190: 153, 243: 177}
OUT = HERE / "loose_history_scan.json"


def main() -> None:
    ort.set_default_logger_severity(4)
    base_hash = {
        task: hashlib.sha256((HERE / "baseline" / f"task{task:03d}.onnx").read_bytes()).hexdigest()
        for task in TASKS
    }
    candidates: list[tuple[int, Path]] = []
    for path in ROOT.rglob("*.onnx"):
        name = path.name.lower()
        for task in TASKS:
            if f"task{task:03d}" in name:
                candidates.append((task, path))
                break
    rows: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()
    for task, path in sorted(candidates, key=lambda item: str(item[1])):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        digest = hashlib.sha256(data).hexdigest()
        if (task, digest) in seen:
            continue
        seen.add((task, digest))
        row: dict[str, object] = {"task": task, "path": str(path.relative_to(ROOT)), "sha256": digest, "baseline_cost": BASE_COST[task]}
        if digest == base_hash[task]:
            row["stage"] = "exact_baseline_duplicate"
            rows.append(row)
            continue
        model, reason, floor = structure_gate(data)
        row["structure_gate"] = reason
        row["static_floor"] = floor
        if model is None or reason != "pass":
            row["stage"] = "structure_reject"
        else:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                screen = actual_screen(data, task)
            row["actual_screen_cost"] = screen
            if screen is None:
                row["stage"] = "unscorable_screen"
            elif screen >= BASE_COST[task]:
                row["stage"] = "screen_not_cheaper"
            else:
                try:
                    with tempfile.TemporaryDirectory(prefix=f"a14_hist_{task:03d}_") as workdir:
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            result = scoring.score_and_verify(
                                copy.deepcopy(model), task, workdir,
                                label=f"a14_hist_{digest[:8]}", require_correct=False,
                            )
                    row["profile"] = result
                    if result is None:
                        row["stage"] = "unscorable_full"
                    elif not result["correct"]:
                        row["stage"] = "known_reject"
                    elif result["cost"] >= BASE_COST[task]:
                        row["stage"] = "full_not_cheaper"
                    else:
                        row["stage"] = "pending_candidate"
                except Exception as exc:  # noqa: BLE001
                    row["stage"] = "profile_error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"SCANNED {len(rows)} unique", flush=True)
            OUT.write_text(json.dumps({"rows": rows, "complete": False}, indent=2) + "\n")
    summary: dict[str, dict[str, int]] = {}
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        counts: dict[str, int] = {}
        for row in task_rows:
            counts[str(row["stage"])] = counts.get(str(row["stage"]), 0) + 1
        summary[str(task)] = {"unique": len(task_rows), **counts}
    pending = [row for row in rows if row["stage"] == "pending_candidate"]
    OUT.write_text(json.dumps({"summary": summary, "rows": rows, "pending": pending, "complete": True}, indent=2) + "\n")
    print(f"DONE unique={len(rows)} pending={len(pending)}", flush=True)


if __name__ == "__main__":
    main()
