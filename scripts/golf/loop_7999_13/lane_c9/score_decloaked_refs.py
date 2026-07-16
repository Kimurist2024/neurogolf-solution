#!/usr/bin/env python3
"""Official-like profile of de-cloaked C9 reference graphs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TASKS = (330, 354, 361, 364)


def main() -> None:
    ort.set_default_logger_severity(4)
    results: dict[str, object] = {}
    for task in TASKS:
        path = HERE / f"task{task:03d}_decloaked.onnx"
        model = onnx.load(path)
        record: dict[str, object] = {}
        try:
            onnx.checker.check_model(model, full_check=True)
            record["full_check"] = True
        except Exception as exc:  # noqa: BLE001
            record["full_check"] = False
            record["full_check_error"] = repr(exc)
        try:
            onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
            record["strict_shape_data_prop"] = True
        except Exception as exc:  # noqa: BLE001
            record["strict_shape_data_prop"] = False
            record["strict_shape_error"] = repr(exc)
        try:
            record["official_like_score"] = scoring.score_and_verify(
                model,
                task,
                str(HERE / "decloaked_score_work"),
                f"decloaked{task}",
                require_correct=False,
            )
        except Exception as exc:  # noqa: BLE001
            record["official_like_score"] = None
            record["score_error"] = repr(exc)
        results[str(task)] = record
        print(task, json.dumps(record, sort_keys=True), flush=True)
        (HERE / "decloaked_audit.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
