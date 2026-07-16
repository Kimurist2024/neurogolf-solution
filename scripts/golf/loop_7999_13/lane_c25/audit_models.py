#!/usr/bin/env python3
"""C25 safety audit for task131/task251 models."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDITOR = HERE.parent / "lane_c11" / "audit_candidates.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location("c11_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    auditor = load_auditor()
    manifest_path = HERE / "models_to_audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output: dict[str, object] = {}
    output_path = HERE / "model_audit.json"
    for item in manifest:
        label = item["label"]
        path = HERE / item["path"]
        output[label] = auditor.audit(label, int(item["task"]), path)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        record = output[label]
        score = record.get("official_like_score") or {}
        trace = record.get("runtime_shape_trace") or {}
        print(
            label,
            "cost=", score.get("cost"),
            "correct=", score.get("correct"),
            "shape_mismatches=", len(trace.get("declared_actual_mismatches", [])),
            flush=True,
        )


if __name__ == "__main__":
    main()
