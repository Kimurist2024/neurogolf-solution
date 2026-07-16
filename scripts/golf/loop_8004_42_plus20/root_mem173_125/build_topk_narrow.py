#!/usr/bin/env python3
"""Build task173 score-ranking dtype probes without changing score order."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from lib import scoring  # noqa: E402
SOURCE = HERE / "task173.onnx"
EXPECTED = "a23d2448c52fe24e949b7758aa754feddfb93012b430fbb1dec10c3e5ce183bf"
VARIANTS = {
    "int8": (TensorProto.INT8, np.int8),
    "uint8": (TensorProto.UINT8, np.uint8),
    "int16": (TensorProto.INT16, np.int16),
    "uint16": (TensorProto.UINT16, np.uint16),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    source = SOURCE.read_bytes()
    if sha(source) != EXPECTED:
        raise RuntimeError(f"authority changed: {sha(source)}")
    rows = []
    for label, (elem_type, dtype) in VARIANTS.items():
        model = onnx.load_from_string(source)
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(0, dtype=dtype), f"topk_{label}_zero"))
        for node in model.graph.node:
            if node.output and node.output[0] == "grid_score":
                node.input[1] = f"topk_{label}_zero"
            if node.output and node.output[0] in {"st_center_score", "st_anchor_score"}:
                for attr in node.attribute:
                    if attr.name == "to":
                        attr.i = elem_type
        changed = {
            "grid_score", "__oh_pre_pix_vals", "__ch_h_12",
            "st_center_score", "st_center_vals", "st_anchor_score", "st_anchor_vals",
        }
        for value in list(model.graph.value_info) + list(model.graph.output):
            if value.name in changed:
                value.type.tensor_type.elem_type = elem_type
        row = {"label": label, "elem_type": elem_type}
        try:
            onnx.checker.check_model(model, full_check=True)
            onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
            out = HERE / f"task173_topk_{label}.onnx"
            onnx.save(model, out)
            row["sha256"] = sha(out.read_bytes())
            for mode, level in (("disable", ort.GraphOptimizationLevel.ORT_DISABLE_ALL), ("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)):
                options = ort.SessionOptions()
                options.graph_optimization_level = level
                try:
                    sanitized = scoring.sanitize_model(onnx.load(out))
                    if sanitized is None:
                        raise RuntimeError("sanitize_model returned None")
                    ort.InferenceSession(sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"])
                    row[mode] = "pass"
                except Exception as exc:
                    row[mode] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            row["checker"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    (HERE / "topk_narrow_build.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
