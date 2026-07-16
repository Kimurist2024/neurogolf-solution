"""Trace task245 log-coordinate intermediates on generator-valid inputs."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[4]
TASKS = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(TASKS))

import task_a1570a43 as task  # noqa: E402


AUTHORITY = Path("/tmp/root_task245_196/task245.onnx")
TRACED = Path("/tmp/root_task245_196/task245_trace.onnx")
NAMES = ("rr_log", "rc_log", "gr_log", "gc_log")


def onehot(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[0, int(color), row, col] = 1.0
    return result


def main() -> None:
    model = copy.deepcopy(onnx.load(AUTHORITY))
    del model.graph.output[:]
    model.graph.output.extend(
        helper.make_tensor_value_info(name, TensorProto.FLOAT16, [])
        for name in NAMES
    )
    onnx.save(model, TRACED)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(TRACED), options, providers=["CPUExecutionProvider"]
    )
    values = {name: [] for name in NAMES}
    for seed in range(10_000):
        random.seed(245_000_000 + seed)
        example = task.generate()
        outputs = session.run(None, {"input": onehot(example["input"])})
        for name, output in zip(NAMES, outputs, strict=True):
            values[name].append(float(np.asarray(output)))

    for name in NAMES:
        array = np.asarray(values[name])
        print(
            name,
            "min=", array.min(),
            "max=", array.max(),
            "negative=", int(np.count_nonzero(array < 0)),
            "zero=", int(np.count_nonzero(array == 0)),
            "positive=", int(np.count_nonzero(array > 0)),
            "unique=", sorted(set(array.tolist())),
        )


if __name__ == "__main__":
    main()
