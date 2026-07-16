"""Trace task319 max-absolute/log source on generator-valid inputs."""

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
MODEL = Path("/private/tmp/ng800946_rank/task319.onnx")
SHARED = ROOT / "scripts/golf/loop_8004_42_plus20/root_selu_scan_127/audit_candidates.py"
NAMES = ("max_abs_f16", "log_abs_f32")


def shared_module():
    spec = importlib.util.spec_from_file_location("selu_audit_shared_199", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load audit helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    shared = shared_module()
    model = onnx.load(MODEL)
    typed = {value.name: value for value in model.graph.value_info}
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    traced.graph.output.extend(copy.deepcopy(typed[name]) for name in NAMES)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    cases, attempts = shared.generate(319, 319_199_001, 10_000)
    values = {name: [] for name in NAMES}
    errors = 0
    for example in cases:
        benchmark = shared.scoring.convert_to_numpy(example)
        try:
            outputs = session.run(NAMES, {session.get_inputs()[0].name: benchmark["input"]})
        except Exception:  # noqa: BLE001
            errors += 1
            continue
        for name, output in zip(NAMES, outputs, strict=True):
            values[name].extend(np.asarray(output).reshape(-1).astype(np.float64).tolist())
    print("cases", len(cases), "attempts", attempts, "errors", errors)
    for name in NAMES:
        array = np.asarray(values[name])
        print(
            name,
            "min", float(array.min()),
            "max", float(array.max()),
            "negative", int(np.count_nonzero(array < 0)),
            "zero", int(np.count_nonzero(array == 0)),
            "nonfinite", int(array.size - np.count_nonzero(np.isfinite(array))),
            "unique", sorted(set(array.tolist())),
        )


if __name__ == "__main__":
    main()
