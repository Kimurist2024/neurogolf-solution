#!/usr/bin/env python3
"""Run the shared strict structure/runtime-shape audit for A28 models."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDITOR = HERE.parent / "lane_c11" / "audit_candidates.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location("a28_shared_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    auditor = load_auditor()
    items = json.loads((HERE / "models_to_audit.json").read_text(encoding="utf-8"))
    output = HERE / "evidence" / "model_audit.json"
    result: dict[str, object] = (
        json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    )
    for item in items:
        label = item["label"]
        if label in result:
            continue
        result[label] = auditor.audit(label, int(item["task"]), HERE / item["path"])
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        row = result[label]
        score = row.get("official_like_score") or {}
        trace = row.get("runtime_shape_trace") or {}
        print(
            label,
            "cost=", score.get("cost"),
            "correct=", score.get("correct"),
            "shape_mismatches=", len(trace.get("declared_actual_mismatches", [])),
            flush=True,
        )


if __name__ == "__main__":
    main()
