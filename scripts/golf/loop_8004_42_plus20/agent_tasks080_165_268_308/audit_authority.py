#!/usr/bin/env python3
"""Non-promoting authority profile for tasks 080/165/268/308."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import uuid
import zipfile
from collections import Counter
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (80, 165, 268, 308)
AUTHORITY_COSTS = {80: 3050, 165: 587, 268: 420, 308: 433}

sys.path.insert(0, str(ROOT / "scripts"))
from golf import check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def first_benchmark(task: int):
    examples = scoring.load_examples(task)
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is not None:
                return benchmark
    raise RuntimeError(f"task{task:03d}: no scorer-valid known example")


def competition_profile(task: int, path: Path) -> dict[str, object]:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize_model rejected authority")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.profile_file_prefix = str(HERE / "evidence" / f"task{task:03d}_{uuid.uuid4().hex[:8]}")
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    benchmark = first_benchmark(task)
    output = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: benchmark["input"]},
    )[0]
    trace = session.end_profiling()
    try:
        memory, params = scoring.score_network(model, trace)
    finally:
        Path(trace).unlink(missing_ok=True)
    if memory is None or params is None:
        raise RuntimeError("official scorer returned None")
    return {
        "memory": int(memory),
        "params": int(params),
        "cost": int(memory + params),
        "runtime_output_shape": list(output.shape),
        "nonfinite": int((~__import__("numpy").isfinite(output)).sum()),
    }


def audit(task: int, data: bytes) -> dict[str, object]:
    path = HERE / "baseline" / f"task{task:03d}.onnx"
    path.write_bytes(data)
    model = onnx.load_model_from_string(data)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    profile = competition_profile(task, path)
    initializer_names = {item.name for item in model.graph.initializer}
    consumers = {name for node in model.graph.node for name in node.input if name}
    produced = {name for node in model.graph.node for name in node.output if name}
    graph_outputs = {item.name for item in model.graph.output}
    return {
        "sha256": sha256(data),
        "bytes": len(data),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "profile": profile,
        "profile_matches_authority": profile["cost"] == AUTHORITY_COSTS[task],
        "checker_full": True,
        "strict_shape_inference_data_prop": True,
        "declared_output_shapes": [dims(item) for item in model.graph.output],
        "inferred_output_shapes": [dims(item) for item in inferred.graph.output],
        "op_histogram": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "standard_domains": all(item.domain in {"", "ai.onnx"} for item in model.opset_import)
        and all(node.domain in {"", "ai.onnx"} for node in model.graph.node),
        "banned_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type.upper()
            in {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
            or "Sequence" in node.op_type
        ],
        "lookup_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type in {"TfIdfVectorizer", "Hardmax"}
        ],
        "center_crop_pad_count": sum(node.op_type == "CenterCropPad" for node in model.graph.node),
        "conv_bias_ub": [list(item) for item in check_conv_bias.check_model(model)],
        "dead_outputs": sorted(produced - consumers - graph_outputs),
        "unused_initializers": sorted(initializer_names - consumers),
    }


def main() -> int:
    (HERE / "baseline").mkdir(parents=True, exist_ok=True)
    (HERE / "evidence").mkdir(parents=True, exist_ok=True)
    zip_path = ROOT / "submission.zip"
    report: dict[str, object] = {
        "authority": "submission.zip (LB 8009.46)",
        "authority_sha256": sha256(zip_path.read_bytes()),
        "all_scores_sha256": sha256((ROOT / "all_scores.csv").read_bytes()),
        "tasks": {},
    }
    with zipfile.ZipFile(zip_path) as archive:
        for task in TASKS:
            data = archive.read(f"task{task:03d}.onnx")
            report["tasks"][str(task)] = audit(task, data)
            print(
                f"task{task:03d}: cost={report['tasks'][str(task)]['profile']['cost']}",
                flush=True,
            )
    output = HERE / "authority_profile.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
