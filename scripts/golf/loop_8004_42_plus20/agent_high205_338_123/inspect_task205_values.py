#!/usr/bin/env python3
"""Inspect the small quantized tensors in the immutable task205 member."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


NAMES = (
    "tall_f", "colq_scale", "roww_max", "roww_thr", "wcolor_counts",
    "counts_i", "box_scaled", "wvec", "code",
)


def summary(array: np.ndarray) -> dict[str, object]:
    values, counts = np.unique(array, return_counts=True)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "unique": values.tolist() if len(values) <= 30 else values[:30].tolist(),
        "unique_counts": counts.tolist() if len(values) <= 30 else counts[:30].tolist(),
        "unique_total": len(values),
    }


def main() -> int:
    model = onnx.load(HERE / "current/task205.onnx")
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    for name in NAMES:
        traced.graph.output.append(copy.deepcopy(typed[name]))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    report: list[dict[str, object]] = []
    examples = scoring.load_examples(205)
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split][:5]):
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            arrays = session.run(list(NAMES), {"input": benchmark["input"]})
            report.append({
                "split": split,
                "index": index,
                "values": {name: summary(np.asarray(array)) for name, array in zip(NAMES, arrays)},
            })
    (HERE / "task205_values.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report:
        print(row["split"], row["index"], {name: row["values"][name]["unique"] for name in ("counts_i", "box_scaled", "wvec")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
