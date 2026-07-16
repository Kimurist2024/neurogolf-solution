#!/usr/bin/env python3
"""Record early-gate failures for every locally cheaper B11 alternative."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from audit_shape_safety import one_case  # noqa: E402


TARGETS = {
    "task264_static358": (
        264,
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task264_r01_static358.onnx",
    ),
    "task376_uint8_gather": (
        376,
        ROOT / "scripts/golf/scratch_codex/task376/cand_uint8_gather.onnx",
    ),
    "task392_seq7": (392, ROOT / "scripts/golf/scratch_codex/task392/task392_seq7.onnx"),
    "task392_drop0": (
        392,
        ROOT / "scripts/golf/scratch_codex/task392/task392_seq7_drop0.onnx",
    ),
    "task392_drop2": (
        392,
        ROOT / "scripts/golf/scratch_codex/task392/task392_seq7_drop2.onnx",
    ),
    "task392_drop5": (
        392,
        ROOT / "scripts/golf/scratch_codex/task392/task392_seq7_drop5.onnx",
    ),
    "task392_drop6": (
        392,
        ROOT / "scripts/golf/scratch_codex/task392/task392_seq7_drop6.onnx",
    ),
}


def main() -> None:
    output: dict[str, object] = {}
    for label, (task, path) in TARGETS.items():
        model = onnx.load(path)
        row: dict[str, object] = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "giant_einsum": any(
                node.op_type == "Einsum" and len(node.input) >= 8
                for node in model.graph.node
            ),
            "lookup": any(node.op_type == "TfIdfVectorizer" for node in model.graph.node),
        }
        try:
            onnx.checker.check_model(model, full_check=True)
            row["checker_full"] = True
        except Exception as exc:  # noqa: BLE001
            row.update(checker_full=False, checker_error=f"{type(exc).__name__}: {exc}")
        try:
            onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
            row["strict_shape_inference"] = True
        except Exception as exc:  # noqa: BLE001
            row.update(
                strict_shape_inference=False,
                strict_shape_error=f"{type(exc).__name__}: {exc}",
            )
        row["disable_all_train_first"] = one_case(task, model, True)
        row["default_train_first"] = one_case(task, model, False)
        if not row.get("checker_full"):
            row["rejection"] = "full checker failure"
        elif row["lookup"]:
            row["rejection"] = "lookup representation prohibited; also fails first known case"
        elif not row["disable_all_train_first"].get("correct", False):
            row["rejection"] = "ORT_DISABLE_ALL first-known-case failure"
        elif not row["default_train_first"].get("correct", False):
            row["rejection"] = "default-ORT first-known-case failure"
        else:
            row["rejection"] = "not rejected by this early audit"
        output[label] = row
        print(label, row["rejection"], flush=True)
    (HERE / "candidate_rejections.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
