#!/usr/bin/env python3
"""Re-audit retained generator-rule controls against the current lane policy."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONTROLS = {
    "task044_rule_autocorr": (44, ROOT / "scripts/golf/loop_7999_13/lane_a23/rule_references/task044_rule_autocorr.onnx"),
    "task117_truthful_copy_hist": (117, ROOT / "scripts/golf/loop_7999_13/lane_a25/rule_references/task117_truthful_copy_hist.onnx"),
    "task330_truthful_component_rect": (330, ROOT / "scripts/golf/loop_7999_13/lane_a26/rule_references/task330_truthful_component_rect.onnx"),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    auditor = load_module(
        "lane153_control_auditor",
        ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
    )
    rows = {}
    for label, (task, path) in CONTROLS.items():
        rows[label] = auditor.audit(label, task, path)
        (HERE / "control_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(label, rows[label].get("official_like_score"), flush=True)


if __name__ == "__main__":
    main()
