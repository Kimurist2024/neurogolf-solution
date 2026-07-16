"""Trace every task366 log2 source on scoreable generator-valid inputs."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL = ROOT / "others/71407/task366.onnx"
SHARED = ROOT / "scripts/golf/loop_8004_42_plus20/root_selu_scan_127/audit_candidates.py"


def load_shared():
    spec = importlib.util.spec_from_file_location("selu_audit_shared_198", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load shared audit")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    shared = load_shared()
    model = onnx.load(MODEL)
    typed = {
        value.name: value
        for value in list(model.graph.value_info) + list(model.graph.output)
    }
    names = [
        node.input[0]
        for node in model.graph.node
        if node.op_type == "Div" and len(node.input) == 2 and node.input[1] == "safe_name_6"
    ]
    if len(names) != 21 or len(set(names)) != 21:
        raise RuntimeError(f"expected 21 unique log sources, got {len(names)}")
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    traced.graph.output.extend(copy.deepcopy(typed[name]) for name in names)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    cases, attempts = shared.generate(366, 366_198_001, 4_000)
    values = {name: [] for name in names}
    runtime_errors = 0
    for example in cases:
        benchmark = shared.scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError("shared.generate returned an unscoreable case")
        try:
            outputs = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
        except Exception:  # noqa: BLE001 - inherited authority failures are counted
            runtime_errors += 1
            continue
        for name, output in zip(names, outputs, strict=True):
            values[name].append(float(np.asarray(output)))
    print("scoreable", len(cases), "attempts", attempts, "runtime_errors", runtime_errors)
    for name in names:
        array = np.asarray(values[name])
        print(
            name,
            "min", float(array.min()),
            "max", float(array.max()),
            "negative", int(np.count_nonzero(array < 0)),
            "zero", int(np.count_nonzero(array == 0)),
            "nonfinite", int(array.size - np.count_nonzero(np.isfinite(array))),
        )


if __name__ == "__main__":
    main()
