#!/usr/bin/env python3
"""Run independent known/random differential validation for harvest finalists."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VALIDATOR = ROOT / "others/7907/neurogolf_team_validator_v1/ngolf_validator.py"
BASE = ROOT / "submission_base_7999.13.zip"
MANIFEST = HERE / "known_winner_manifest.json"

# task219 is explicitly confirmed private-zero in docs/golf/private_zero_tasks.md
# and must not be reintroduced even though its visible gold/cost look excellent.
POST_HARVEST_EXCLUDED = {219}


def main() -> int:
    winners = json.loads(MANIFEST.read_text())["winners"]
    jobs = []
    for task_text, winner in sorted(winners.items(), key=lambda item: int(item[0])):
        task = int(task_text)
        if task in POST_HARVEST_EXCLUDED:
            continue
        output = HERE / f"external_task{task:03d}.json"
        log = HERE / f"external_task{task:03d}.log"
        command = [
            sys.executable,
            str(VALIDATOR),
            "validate-task",
            "--task",
            str(task),
            "--candidate-model",
            str(ROOT / winner["candidate"]),
            "--baseline-zip",
            str(BASE),
            "--data-dir",
            str(ROOT / "inputs/neurogolf-2026"),
            "--data-zip",
            str(ROOT / "inputs/neurogolf-2026.zip"),
            "--random-cases",
            "3000",
            "--seed",
            str(799_913_000 + task),
            "--out-json",
            str(output),
        ]
        jobs.append({"task": task, "command": command, "output": output, "log": log})

    pending = iter(jobs)
    active = []
    exhausted = False
    results = []
    while active or not exhausted:
        while len(active) < 3 and not exhausted:
            try:
                job = next(pending)
            except StopIteration:
                exhausted = True
                break
            handle = job["log"].open("wb")
            process = subprocess.Popen(job["command"], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            active.append({**job, "process": process, "handle": handle, "start": time.monotonic()})
        for slot in list(active):
            process = slot["process"]
            elapsed = time.monotonic() - slot["start"]
            status = process.poll()
            if status is None and elapsed < 360:
                continue
            if status is None:
                process.kill()
                status = process.wait()
                timed_out = True
            else:
                timed_out = False
            slot["handle"].close()
            result = {
                "task": slot["task"],
                "returncode": status,
                "timeout": timed_out,
                "json": str(slot["output"].relative_to(ROOT)),
                "log": str(slot["log"].relative_to(ROOT)),
            }
            if slot["output"].exists():
                payload = json.loads(slot["output"].read_text())
                result["verdict"] = payload.get("decision", {}).get("verdict")
                result["decision"] = payload.get("decision")
                result["differential"] = payload.get("differential")
            results.append(result)
            active.remove(slot)
            print(
                f"task{slot['task']:03d} rc={status} verdict={result.get('verdict')} timeout={timed_out}",
                flush=True,
            )
        if active:
            time.sleep(0.1)

    (HERE / "external_validation_summary.json").write_text(
        json.dumps(
            {
                "baseline_zip": str(BASE.relative_to(ROOT)),
                "random_cases": 3000,
                "post_harvest_excluded": sorted(POST_HARVEST_EXCLUDED),
                "results": sorted(results, key=lambda result: result["task"]),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
