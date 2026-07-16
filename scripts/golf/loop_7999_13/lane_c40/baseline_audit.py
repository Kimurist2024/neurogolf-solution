#!/usr/bin/env python3
"""Independently audit the authoritative task391 baseline member."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import onnxruntime as ort


HERE = Path(__file__).resolve().parent
AUDITOR = HERE.parent / "lane_c11" / "audit_candidates.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("c40_c11_auditor", AUDITOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ort.set_default_logger_severity(4)
    row = module.audit(
        "lane_c40_authoritative_baseline_task391",
        391,
        HERE / "baseline" / "task391.onnx",
    )
    (HERE / "baseline_audit.json").write_text(
        json.dumps(row, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sha256": row["sha256"],
                "score": row["official_like_score"],
                "known_disable_all": row["known_disable_all"]["total"],
                "known_default": row["known_default"]["total"],
                "runtime_shape_trace": row["runtime_shape_trace"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
