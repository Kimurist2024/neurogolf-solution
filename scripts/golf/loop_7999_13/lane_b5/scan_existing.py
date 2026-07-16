#!/usr/bin/env python3
"""Score byte-distinct, lane-targeted historical ONNX candidates read-only."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TASKS = (99, 239, 268, 297, 345, 374, 394)
BASE_COST = {99: 398, 239: 384, 268: 446, 297: 371, 345: 389, 374: 481, 394: 350}


def main() -> None:
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
            with tempfile.TemporaryDirectory(prefix=f"b5_scan_{task:03d}_") as workdir:
                # Historical shape-cloak rejects can emit thousands of ORT warnings.
                # Keep the durable audit JSON concise while retaining the exception.
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = scoring.score_and_verify(
                        onnx.load(path), task, workdir, label="b5", require_correct=False
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
        except Exception as exc:  # historical rejects are expected
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    output = Path(__file__).with_name("existing_scan.json")
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    winners = [r for r in rows if r.get("correct") and r.get("cheaper")]
    print("WINNERS", json.dumps(winners, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
