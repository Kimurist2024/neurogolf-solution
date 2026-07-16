#!/usr/bin/env python3
"""Run the established strict structural/cost/known audit on A34 models."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDITOR = HERE.parent / "lane_c11" / "audit_candidates.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location("a34_c11_auditor", AUDITOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    auditor = load_auditor()
    rows = {
        "baseline_task398": auditor.audit("baseline_task398", 398, HERE / "baseline_task398.onnx"),
        "task398_q4_d_reuse_347": auditor.audit(
            "task398_q4_d_reuse_347", 398, HERE / "task398_q4_d_reuse_347.onnx"
        ),
    }
    (HERE / "strict_audit.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for label, row in rows.items():
        print(label, row.get("official_like_score"))
