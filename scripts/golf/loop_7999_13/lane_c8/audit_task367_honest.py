#!/usr/bin/env python3
"""Audit the closest task367 graph after removing every shape cloak."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort

import audit_sound_references as audit


HERE = Path(__file__).resolve().parent


def main() -> None:
    ort.set_default_logger_severity(4)
    path = HERE / "sound_references/task367_no_shape_cloak.onnx"
    with tempfile.TemporaryDirectory(prefix="c8_367_honest_", dir="/tmp") as workdir:
        score = audit.scoring.score_and_verify(
            onnx.load(path), 367, workdir, label="c8honest", require_correct=False
        )
    disabled = audit.session(path, True)
    default = audit.session(path, False)
    result = {
        "path": str(path.relative_to(audit.ROOT)),
        "score": score,
        "known_disable_all": audit.known(367, disabled),
        "fresh_disable_all": audit.fresh(367, "e73095fd", disabled, True),
        "fresh_default_ort": audit.fresh(367, "e73095fd", default, False),
        "center_crop_pad_nodes": sum(
            node.op_type == "CenterCropPad" for node in onnx.load(path).graph.node
        ),
    }
    (HERE / "task367_honest_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(result)


if __name__ == "__main__":
    main()
