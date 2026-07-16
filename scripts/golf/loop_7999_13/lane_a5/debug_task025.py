#!/usr/bin/env python3
"""Compare selected task025 intermediates for rejected algebra experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


NAMES = [
    "vcolg", "vleftq_58", "vlineq_44", "vpright_63", "vpleft_64",
    "vMrq_66", "vMlq_68", "hcolg", "hleftq_98", "hlineq_84",
    "hpright_103", "hpleft_104", "hMrq_106", "hMlq_108", "Nsumq_112", "output",
]


def run(path: Path, x: np.ndarray) -> dict[str, np.ndarray]:
    model = onnx.load(path)
    existing = {value.name for value in model.graph.output}
    for name in NAMES:
        if name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.DOUBLE, None))
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), opts)
    values = session.run(NAMES, {"input": x})
    return dict(zip(NAMES, values, strict=True))


def main() -> None:
    example = scoring.load_examples(25)["train"][0]
    x = scoring.convert_to_numpy(example)["input"]
    here = Path(__file__).resolve().parent
    base = run(here / "baseline" / "task025.onnx", x)
    for candidate_name in ("task025_drop_sigv.onnx", "task025_drop_negsigk.onnx"):
        candidate = run(here / "candidates" / candidate_name, x)
        print(candidate_name)
        for name in NAMES:
            a, b = base[name], candidate[name]
            same = np.array_equal(a, b)
            nz = np.abs(a) > 0
            ratio = b[nz] / a[nz] if np.any(nz) else np.array([])
            ratio_text = "-" if ratio.size == 0 else f"[{ratio.min():.8g},{ratio.max():.8g}]"
            print(
                f"{name:14s} same={same!s:5s} base=[{a.min():.8g},{a.max():.8g}] "
                f"cand=[{b.min():.8g},{b.max():.8g}] ratio={ratio_text}"
            )


if __name__ == "__main__":
    main()
