#!/usr/bin/env python3
"""Strict read-only model audit for the target-mid18 lane.

All reports are written beside this script.  Shared/root models are read only;
no candidate is promoted and no ZIP is created.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load_auditor():
    source = ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py"
    spec = importlib.util.spec_from_file_location("mid18_shared_auditor", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    auditor = load_auditor()
    base = ROOT / "scripts/golf/loop_8004_42_plus20/agent_exact_wave2/base"
    archive = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400"
    models: dict[str, tuple[int, Path]] = {
        "baseline_task099": (99, base / "task099.onnx"),
        "baseline_task279": (279, base / "task279.onnx"),
        "baseline_task345": (345, base / "task345.onnx"),
        "baseline_task239": (239, base / "task239.onnx"),
        "baseline_task075": (75, base / "task075.onnx"),
        "baseline_task392": (392, base / "task392.onnx"),
        "baseline_task387": (387, base / "task387.onnx"),
        "baseline_task225": (225, base / "task225.onnx"),
        "history_task099_cost397_r01": (99, archive / "task099_r01_static397.onnx"),
        "history_task099_cost397_r02": (99, archive / "task099_r02_static397.onnx"),
        "history_task099_cost397_r03": (99, archive / "task099_r03_static397.onnx"),
        "history_task099_cost397_r04": (99, archive / "task099_r04_static397.onnx"),
        "history_task279_cost357_r01": (279, archive / "task279_r01_static357.onnx"),
        "history_task279_cost357_r02": (279, archive / "task279_r02_static357.onnx"),
        "history_task279_cost357_r03": (279, archive / "task279_r03_static357.onnx"),
        "history_task279_cost358_r04": (279, archive / "task279_r04_static358.onnx"),
        "history_task279_cost358_r05": (279, archive / "task279_r05_static358.onnx"),
        "history_task345_cost365": (345, archive / "task345_r01_static365.onnx"),
        "history_task239_cost374_r01": (239, archive / "task239_r01_static374.onnx"),
        "history_task239_cost374_r02": (239, archive / "task239_r02_static374.onnx"),
        "history_task239_cost379_r03": (239, archive / "task239_r03_static379.onnx"),
        "history_task392_cost341_r01": (392, archive / "task392_r01_static341.onnx"),
        "history_task392_cost341_r02": (392, archive / "task392_r02_static341.onnx"),
        "history_task392_cost341_r03": (392, archive / "task392_r03_static341.onnx"),
        "history_task392_cost341_r04": (392, archive / "task392_r04_static341.onnx"),
        "history_task392_cost341_r05": (392, archive / "task392_r05_static341.onnx"),
        "history_task225_cost306_r01": (225, archive / "task225_r01_static306.onnx"),
        "history_task225_cost306_r02": (225, archive / "task225_r02_static306.onnx"),
        "probe_task387_static_size": (
            387,
            ROOT
            / "scripts/golf/loop_8004_42_plus20/agent_exact_size16/candidates/task387.onnx",
        ),
        "control_task345_legal_same_cost389": (
            345,
            ROOT / "scripts/golf/loop_7999_13/lane_b30/task345_legal_swapped_prescaled_cost389.onnx",
        ),
        "probe_task239_zero_blank": (239, HERE / "task239_zero_blank.onnx"),
    }
    report: dict[str, object] = {}
    out = HERE / "model_audit.json"
    for label, (task, path) in models.items():
        if not path.is_file():
            report[label] = {"task": task, "path": str(path), "missing": True}
        else:
            report[label] = auditor.audit(label, task, path)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        row = report[label]
        profile = row.get("official_like_score") if isinstance(row, dict) else None
        print(
            label,
            None if not isinstance(profile, dict) else profile.get("cost"),
            None if not isinstance(profile, dict) else profile.get("correct"),
            flush=True,
        )


if __name__ == "__main__":
    main()
