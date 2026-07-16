#!/usr/bin/env python3
"""Repair task366 candidate metadata from observed runtime shapes without changing computation."""

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
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from lib import scoring  # noqa: E402


SOURCE = ROOT / "others/2/1201/7120/task366_further_improved.onnx"
BASELINE_ZIP = ROOT / "submission_base_8000.46.zip"
BASELINE = HERE / "baseline_task366.onnx"
SOURCE_COPY = HERE / "source_task366_cost7646_shape_cloak.onnx"
REPAIRED = HERE / "task366_cost7646_truthful_metadata.onnx"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_trace_session(model: onnx.ModelProto):
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=False, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return session, names


def trace(session, names: list[str], benchmark_input: np.ndarray) -> dict[str, np.ndarray]:
    values = session.run(names, {session.get_inputs()[0].name: benchmark_input})
    return {name: np.asarray(value) for name, value in zip(names, values)}


def computational_fingerprint(model: onnx.ModelProto) -> str:
    payload = copy.deepcopy(model)
    del payload.graph.value_info[:]
    for value in payload.graph.input:
        value.type.ClearField("tensor_type")
    for value in payload.graph.output:
        value.type.ClearField("tensor_type")
    return hashlib.sha256(payload.SerializeToString(deterministic=True)).hexdigest()


def main() -> int:
    with zipfile.ZipFile(BASELINE_ZIP) as archive:
        BASELINE.write_bytes(archive.read("task366.onnx"))
    SOURCE_COPY.write_bytes(SOURCE.read_bytes())

    source = onnx.load(SOURCE)
    session, names = make_trace_session(source)
    examples = scoring.load_examples(366)
    executable_arc = [
        example
        for example in examples["arc-gen"]
        if scoring.convert_to_numpy(example) is not None
    ]
    probes = [
        ("train0", examples["train"][0]),
        ("train_last", examples["train"][-1]),
        ("test0", examples["test"][0]),
        ("arc_gen0", executable_arc[0]),
        ("arc_gen_last_executable", executable_arc[-1]),
    ]
    traces: dict[str, dict[str, np.ndarray]] = {}
    for label, example in probes:
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError(f"unconvertible probe {label}")
        traces[label] = trace(session, names, benchmark["input"])
    first = traces["train0"]
    variable_shapes = {
        name: {
            label: list(values[name].shape)
            for label, values in traces.items()
        }
        for name in names
        if len({tuple(values[name].shape) for values in traces.values()}) > 1
    }

    repaired = copy.deepcopy(source)
    graph_outputs = {value.name for value in repaired.graph.output}
    del repaired.graph.value_info[:]
    declared = 0
    for name in names:
        if name in graph_outputs:
            continue
        value = first[name]
        dtype = helper.np_dtype_to_tensor_dtype(value.dtype)
        repaired.graph.value_info.append(
            helper.make_tensor_value_info(name, dtype, list(value.shape))
        )
        declared += 1
    onnx.checker.check_model(repaired, full_check=True)
    onnx.shape_inference.infer_shapes(
        copy.deepcopy(repaired), strict_mode=True, data_prop=True
    )
    onnx.save(repaired, REPAIRED)

    payload = {
        "baseline_zip": str(BASELINE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": digest(BASELINE_ZIP),
        "baseline_model_sha256": digest(BASELINE),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": digest(SOURCE),
        "repaired": str(REPAIRED.relative_to(ROOT)),
        "repaired_sha256": digest(REPAIRED),
        "source_computational_fingerprint": computational_fingerprint(source),
        "repaired_computational_fingerprint": computational_fingerprint(repaired),
        "node_count": len(source.graph.node),
        "initializer_count": len(source.graph.initializer),
        "source_value_info_count": len(source.graph.value_info),
        "repaired_value_info_count": declared,
        "runtime_tensor_count": len(names),
        "probe_count": len(probes),
        "variable_runtime_shapes": variable_shapes,
        "first_probe_intermediate_bytes": sum(
            int(value.nbytes) for name, value in first.items() if name not in graph_outputs
        ),
    }
    if payload["source_computational_fingerprint"] != payload["repaired_computational_fingerprint"]:
        raise RuntimeError("metadata-only repair changed computation")
    (HERE / "build_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
