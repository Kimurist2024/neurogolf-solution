#!/usr/bin/env python3
"""Score byte-distinct historical candidates for B6 tasks, read-only."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import onnx
import onnxruntime


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TASKS = (75, 159, 200, 218, 225, 228, 388)
BASE_COST = {75: 345, 159: 293, 200: 346, 218: 329, 225: 333, 228: 302, 388: 311}


def main() -> None:
    onnxruntime.set_default_logger_severity(3)
    rows: list[dict[str, object]] = []
    candidates: list[tuple[int, Path]] = []
    for path in ROOT.rglob("*.onnx"):
        text = path.name.lower()
        for task in TASKS:
            if f"task{task:03d}" in text:
                candidates.append((task, path))
                break

    seen: set[tuple[int, str]] = set()
    for task, path in sorted(candidates, key=lambda item: str(item[1])):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        key = (task, digest)
        if key in seen:
            continue
        seen.add(key)
        record: dict[str, object] = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
        }
        try:
            with tempfile.TemporaryDirectory(prefix=f"b6_scan_{task:03d}_") as workdir:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = scoring.score_and_verify(
                        onnx.load(path), task, workdir, label="b6", require_correct=False
                    )
            if result is None:
                record["status"] = "unscorable"
            else:
                record.update(
                    status="ok",
                    cost=int(result["cost"]),
                    memory=int(result["memory"]),
                    params=int(result["params"]),
                    correct=bool(result["correct"]),
                    cheaper=int(result["cost"]) < BASE_COST[task],
                )
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    output = Path(__file__).with_name("existing_scan.json")
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    winners = [row for row in rows if row.get("correct") and row.get("cheaper")]
    print("WINNERS", json.dumps(winners, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
