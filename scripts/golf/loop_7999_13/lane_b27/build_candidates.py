#!/usr/bin/env python3
"""Build non-promoting task382 shape-repair probes for B27."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from lib import scoring  # noqa: E402


SOURCE = ROOT / "scripts/golf/loop_7999_13/lane_headroom/candidates/task382.onnx"
BASELINE_ZIP = ROOT / "submission_base_8000.46.zip"
BASELINE = HERE / "baseline_task382.onnx"
OUTPUT_ONLY = HERE / "task382_output_shape_only.onnx"
HORIZONTAL = HERE / "task382_declared_horizontal_runtime.onnx"
VERTICAL = HERE / "task382_declared_vertical_runtime.onnx"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_shape(value: onnx.ValueInfoProto, shape: tuple[int, ...] | list[int]) -> None:
    tensor_type = value.type.tensor_type
    del tensor_type.shape.dim[:]
    for size in shape:
        tensor_type.shape.dim.add().dim_value = int(size)


def make_session(model: onnx.ModelProto) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options)


def trace_shapes(model: onnx.ModelProto, benchmark_input: np.ndarray) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=False, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    values = make_session(traced).run(names, {"input": benchmark_input})
    return {name: tuple(np.asarray(value).shape) for name, value in zip(names, values)}


def write_declared_runtime(
    source: onnx.ModelProto,
    runtime_shapes: dict[str, tuple[int, ...]],
    destination: Path,
) -> None:
    model = copy.deepcopy(source)
    for value in model.graph.value_info:
        set_shape(value, runtime_shapes[value.name])
    set_shape(model.graph.output[0], runtime_shapes[model.graph.output[0].name])
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    onnx.save(model, destination)


def main() -> int:
    with zipfile.ZipFile(BASELINE_ZIP) as archive:
        baseline_data = archive.read("task382.onnx")
    BASELINE.write_bytes(baseline_data)

    source = onnx.load(SOURCE)
    output_only = copy.deepcopy(source)
    set_shape(output_only.graph.output[0], (1, 10, 30, 30))
    output_only_checker_error = None
    try:
        onnx.checker.check_model(output_only, full_check=True)
        onnx.shape_inference.infer_shapes(
            copy.deepcopy(output_only), strict_mode=True, data_prop=True
        )
    except Exception as exc:  # Expected evidence for the output-only repair.
        output_only_checker_error = f"{type(exc).__name__}: {exc}"
    onnx.save(output_only, OUTPUT_ONLY)

    examples = scoring.load_examples(382)["train"]
    horizontal_input = scoring.convert_to_numpy(examples[0])["input"]
    vertical_input = scoring.convert_to_numpy(examples[2])["input"]
    horizontal_shapes = trace_shapes(source, horizontal_input)
    vertical_shapes = trace_shapes(source, vertical_input)
    if horizontal_shapes["output"] != (1, 10, 30, 30):
        raise RuntimeError(horizontal_shapes["output"])
    if vertical_shapes["output"] != (1, 10, 30, 30):
        raise RuntimeError(vertical_shapes["output"])

    write_declared_runtime(source, horizontal_shapes, HORIZONTAL)
    write_declared_runtime(source, vertical_shapes, VERTICAL)

    orientation_dependent = {
        name: {
            "horizontal": list(horizontal_shapes[name]),
            "vertical": list(vertical_shapes[name]),
        }
        for name in horizontal_shapes
        if horizontal_shapes[name] != vertical_shapes[name]
    }
    payload = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": digest(SOURCE),
        "baseline_zip": str(BASELINE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": digest(BASELINE_ZIP),
        "baseline_model_sha256": digest(BASELINE),
        "artifacts": {
            path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in (OUTPUT_ONLY, HORIZONTAL, VERTICAL)
        },
        "output_only_checker_error": output_only_checker_error,
        "orientation_dependent_runtime_shapes": orientation_dependent,
    }
    (HERE / "build_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
