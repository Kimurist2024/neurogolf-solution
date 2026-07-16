#!/usr/bin/env python3
"""Fresh two-runtime audit for the no-cloak task310 safe-selector reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx
import onnxruntime as ort

from audit_baselines import fresh, known_complete, make_session


HERE = Path(__file__).resolve().parent
PATH = HERE / "task310_safe_linear_selector.onnx"


def main() -> None:
    ort.set_default_logger_severity(4)
    model = onnx.load(PATH)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    disabled = make_session(PATH, True)
    default = make_session(PATH, False)
    record = {
        "path": str(PATH),
        "sha256": hashlib.sha256(PATH.read_bytes()).hexdigest(),
        "full_check": True,
        "strict_shape_data_prop": True,
        "value_info_count": len(model.graph.value_info),
        "known_disable_all": known_complete(310, disabled),
        "known_default_ort": known_complete(310, default),
        "fresh_disable_all": fresh(310, "c909285e", disabled, True),
        "fresh_default_ort": fresh(310, "c909285e", default, False),
    }
    (HERE / "task310_safe_audit.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
