#!/usr/bin/env python3
"""Run two independent dual-ORT generator audits for selected candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HELPERS = ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2"
sys.path.insert(0, str(HELPERS))

from rescreen_all import fresh_dual  # noqa: E402


CANDIDATES = {
    205: ROOT / "scripts/golf/loop_7999_13/lane_a23/candidates/task205_r02.onnx",
    343: ROOT / "scripts/golf/loop_7999_13/lane_c39/candidate/task343.onnx",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--count", type=int, default=5000)
    args = parser.parse_args()

    path = CANDIDATES[args.task]
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    row = {"task": args.task, "sha256": digest, "data": data}
    result = {
        "task": args.task,
        "candidate": str(path.relative_to(ROOT)),
        "sha256": digest,
        "count_per_seed": args.count,
        "seeds": [],
    }
    for seed in (90_700_000 + args.task * 10 + 1, 90_700_000 + args.task * 10 + 2):
        result["seeds"].append(fresh_dual(args.task, [row], args.count, seed))
        output = HERE / f"task{args.task:03d}_fresh_dual_two_seeds.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
