#!/usr/bin/env python3
"""Strict audit of the ten frozen task201-400 baseline members."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SHARED = ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py"
# task333's 36-input floating Einsum takes several minutes even for the known
# corpus.  Its current SHA already has a completed 265/265 + raw-2000 audit in
# loop_8003_40/agent_exact_resume, so this fresh current-baseline pass covers
# the other nine and the final report links that existing task333 evidence.
TASKS = (374, 250, 324, 308, 275, 338, 268, 377, 279)


def load_shared():
    spec = importlib.util.spec_from_file_location("batch20_shared_auditor", SHARED)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    shared = load_shared()
    output_path = HERE / "CURRENT_AUDIT.json"
    output = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else {}
    )
    for task in TASKS:
        key = f"task{task:03d}"
        if key in output:
            continue
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        output[key] = shared.audit(key, task, path)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        score = output[key].get("official_like_score") or {}
        trace = output[key].get("runtime_shape_trace") or {}
        print(
            key,
            f"cost={score.get('cost')}",
            f"known={score.get('correct')}",
            f"shape_mismatches={len(trace.get('declared_actual_mismatches', []))}",
            flush=True,
        )


if __name__ == "__main__":
    main()
