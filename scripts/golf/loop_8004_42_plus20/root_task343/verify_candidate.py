#!/usr/bin/env python3
"""Independent multi-seed dual-ORT audit for the clean task343 candidate."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HELPERS = ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2"
sys.path.insert(0, str(HELPERS))

from rescreen_all import fresh_dual, known_dual  # noqa: E402


CANDIDATE = ROOT / "scripts/golf/loop_7999_13/lane_c39/candidate/task343.onnx"
HERE = Path(__file__).resolve().parent


def main() -> int:
    data = CANDIDATE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    row = {"task": 343, "sha256": digest, "data": data}
    result = {
        "task": 343,
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "sha256": digest,
        "known_dual": known_dual(343, data),
        "fresh_seeds": [],
    }
    for seed in (90_343_001, 90_343_002):
        result["fresh_seeds"].append(fresh_dual(343, [row], 5000, seed))
        (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
