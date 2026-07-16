#!/usr/bin/env python3
"""Audit one high47 model without mutating any shared NeuroGolf artifact.

The caller intentionally runs each model in a separate process.  A few
historical models exercise invalid ORT allocator paths, so process isolation is
part of the fail-closed gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUDITOR = ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location("high47_shared_auditor", AUDITOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import auditor: {AUDITOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--task", required=True, type=int)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    auditor = load_auditor()
    result = auditor.audit(args.label, args.task, args.model.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    score = result.get("official_like_score") or {}
    print(
        json.dumps(
            {
                "label": args.label,
                "task": args.task,
                "cost": score.get("cost"),
                "correct": score.get("correct"),
                "full_check": result.get("full_check"),
                "strict": result.get("strict_shape_data_prop"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
