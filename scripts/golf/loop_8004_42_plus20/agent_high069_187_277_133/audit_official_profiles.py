#!/usr/bin/env python3
"""Measure authority members with competition score_and_verify."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def main() -> int:
    rows = {}
    for task in (69, 187, 277):
        model = onnx.load(HERE / f"current/task{task:03d}.onnx")
        with tempfile.TemporaryDirectory(prefix=f"high133_{task:03d}_", dir="/tmp") as workdir:
            profile = scoring.score_and_verify(
                copy.deepcopy(model), task, workdir,
                label=f"high133_task{task:03d}_authority",
                require_correct=False,
            )
        if profile is None:
            raise RuntimeError(f"task{task:03d}: score_and_verify returned None")
        rows[str(task)] = profile
        print(
            f"task{task:03d} memory={profile['memory']} params={profile['params']} "
            f"cost={profile['cost']} correct={profile.get('correct')}",
            flush=True,
        )
    (HERE / "official_profiles.json").write_text(
        json.dumps({"tasks": rows}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
