#!/usr/bin/env python3
"""Keep the 30-minute NeuroGolf exploration loop alive until explicitly stopped."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "others/71407/continuous_30m"
ROTATION = ROOT / "scripts/golf/continuous_30m_rotation.py"
STOP = False
CHILD: subprocess.Popen[str] | None = None
LOCK_HANDLE = None


def next_cycle() -> int:
    pattern = re.compile(r"cycle_(\d{4})_")
    found = []
    for path in OUT.glob("cycle_*"):
        match = pattern.match(path.name)
        if match:
            found.append(int(match.group(1)))
    return max(found, default=-1) + 1


def record(event: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    event = {
        "utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with (OUT / "supervisor_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def stop_handler(_signum, _frame) -> None:
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.terminate()


def main() -> int:
    global CHILD, LOCK_HANDLE
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--restart-delay", type=int, default=30)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LOCK_HANDLE = (OUT / "supervisor.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(LOCK_HANDLE.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another continuous supervisor already owns the lock", flush=True)
        return 2
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    record({"event": "supervisor_started"})

    while not STOP:
        cycle = next_cycle()
        command = [
            sys.executable,
            str(ROTATION),
            "--interval", str(args.interval),
            "--timeout", str(args.timeout),
            "--start-cycle", str(cycle),
        ]
        record({"event": "rotation_started", "start_cycle": cycle})
        CHILD = subprocess.Popen(command, cwd=ROOT, text=True)
        returncode = CHILD.wait()
        CHILD = None
        record({"event": "rotation_exited", "returncode": returncode})
        if STOP:
            break
        deadline = time.monotonic() + max(1, args.restart_delay)
        while not STOP and time.monotonic() < deadline:
            time.sleep(min(1, deadline - time.monotonic()))

    record({"event": "supervisor_stopped"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
