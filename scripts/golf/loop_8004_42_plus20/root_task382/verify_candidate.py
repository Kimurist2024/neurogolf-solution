#!/usr/bin/env python3
"""Independent multi-seed dual-ORT audit for the task382 rule fix."""

from __future__ import annotations

import hashlib
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HELPERS = ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2"
sys.path.insert(0, str(HELPERS))

from rescreen_all import fresh_dual, known_dual  # noqa: E402


DEFAULT_CANDIDATE = ROOT / "scripts/golf/loop_7999_13/lane_headroom/candidates/task382.onnx"
HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=HERE / "verification.json")
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    data = candidate.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    row = {"task": 382, "sha256": digest, "data": data}
    result = {
        "task": 382,
        "candidate": str(candidate.relative_to(ROOT)),
        "sha256": digest,
        "known_dual": known_dual(382, data),
        "fresh_seeds": [],
    }
    for seed in (88_382_001, 88_382_002):
        result["fresh_seeds"].append(fresh_dual(382, [row], args.count, seed))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
