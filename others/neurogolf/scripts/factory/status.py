#!/usr/bin/env python3
"""Print one-screen status for the detached overnight factory."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FACTORY = REPO_ROOT / "artifacts" / "factory"
STATUS = FACTORY / "status.json"
STATE = FACTORY / "state.json"
HEARTBEAT = FACTORY / "heartbeat"
RESULTS = FACTORY / "results.log"


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _age(path: Path) -> str:
    if not path.exists():
        return "missing"
    seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    return f"{int(seconds)}s ago"


def _recent_results(limit: int = 5) -> list[dict[str, Any]]:
    if not RESULTS.is_file():
        return []
    lines = RESULTS.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"raw": line})
    return out


def main() -> int:
    status = _read_json(STATUS, {})
    state = _read_json(STATE, {})
    active = status.get("active_workers", state.get("in_progress", []))
    queue = status.get("queue", {})
    expected = status.get("expected_score", {})

    print("NeuroGolf factory")
    print(f"heartbeat: {_age(HEARTBEAT)}")
    print(f"driver pid: {status.get('driver_pid', 'unknown')}")
    print(f"paused: {status.get('paused', False)}")
    print(
        "queue: "
        f"{queue.get('remaining', 'unknown')} remaining / "
        f"{queue.get('total', 'unknown')} total, "
        f"{status.get('done_count', len(state.get('done', [])))} done"
    )
    print(
        "workers: "
        f"{len(active)} active / {status.get('workers_configured', 'unknown')} "
        "configured"
    )
    for item in active[:12]:
        task = int(item.get("task", 0))
        print(
            f"  task{task:03d} pid={item.get('pid')} "
            f"cost={item.get('cost')} slot={item.get('slot')}"
        )
    print(
        "promotions: "
        f"{status.get('promotions', state.get('promotions', 0))} total, "
        f"{status.get('promotions_since_last_merge', 0)} since last merge"
    )
    print(
        "merge: "
        f"{status.get('last_merge', {}).get('status', 'unknown')} at "
        f"{status.get('last_merge', {}).get('at')}"
    )
    print(
        "submit: "
        f"{status.get('last_submit', {}).get('status', 'unknown')} at "
        f"{status.get('last_submit', {}).get('at')}"
    )
    print(
        "expected score: "
        f"{expected.get('score')} from {expected.get('source')} "
        f"delta={expected.get('delta')}"
    )
    recent = status.get("recent_results")
    if recent is None:
        recent = _recent_results()
    if recent:
        print("recent results:")
        for row in recent:
            if "raw" in row:
                print(f"  {row['raw']}")
                continue
            print(
                f"  task{int(row['task']):03d} exit={row['exit_code']} "
                f"promoted={row['promoted']} new_cost={row['new_cost']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
