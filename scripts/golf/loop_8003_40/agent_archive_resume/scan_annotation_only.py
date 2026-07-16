#!/usr/bin/env python3
"""Find archive candidates that only correct non-computational value_info.

The 8003.40 baseline is immutable.  A candidate is retained only when clearing
``graph.value_info`` makes its deterministic protobuf byte-for-byte identical
to the corresponding baseline model.  This is deliberately much narrower than
an operator/equivalence heuristic: nodes, attributes, initializers, graph I/O,
opsets, functions, and metadata must already be identical.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE.parent / "agent_archive_rescreen" / "archive_static_scan.json"
BASE_DIR = HERE.parent / "base_models"
OUTPUT = HERE / "annotation_only_scan.json"

# Full quarantine set from docs/golf/private_zero_tasks.md, plus the ambiguous
# black pair, pollution models, and task-level high-risk entries discovered in
# the historical notes.  Annotation-only equality is safe in principle, but a
# candidate that preserves an incumbent UB is still not eligible.
QUARANTINED = {
    9, 15, 18, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 96, 101, 102,
    112, 133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 182,
    185, 187, 191, 192, 196, 198, 202, 204, 205, 208, 209, 216, 219,
    222, 233, 246, 255, 264, 277, 285, 286, 302, 319, 325, 333, 346,
    361, 365, 366, 372, 377, 379, 391, 393, 396,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        else:
            result.append(dim.dim_param)
    return result


def computational_payload(model: onnx.ModelProto) -> bytes:
    clone = onnx.ModelProto()
    clone.CopyFrom(model)
    del clone.graph.value_info[:]
    return clone.SerializeToString(deterministic=True)


def bias_ub(model: onnx.ModelProto) -> list[dict[str, object]]:
    """Check Conv-family static bias length against inferred output channels."""
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    out_channels: dict[str, int] = {}
    for value in list(inferred.graph.value_info) + list(inferred.graph.output):
        shape = value.type.tensor_type.shape
        if len(shape.dim) >= 2 and shape.dim[1].HasField("dim_value"):
            channels = int(shape.dim[1].dim_value)
            if channels > 0:
                out_channels[value.name] = channels
    initializers = {item.name: item for item in model.graph.initializer}
    failures: list[dict[str, object]] = []
    for node in model.graph.node:
        index = 8 if node.op_type == "QLinearConv" else (
            2 if node.op_type in {"Conv", "ConvTranspose"} else None
        )
        if index is None or len(node.input) <= index or not node.input[index]:
            continue
        bias = initializers.get(node.input[index])
        channels = out_channels.get(node.output[0])
        if bias is None or channels is None:
            continue
        length = int(numpy_helper.to_array(bias).size)
        if length != channels:
            failures.append({
                "op": node.op_type,
                "node": node.name,
                "bias": bias.name,
                "bias_length": length,
                "output_channels": channels,
            })
    return failures


def static_cost(model: onnx.ModelProto) -> int:
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    infos = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    excluded = {item.name for item in inferred.graph.input}
    excluded.update(item.name for item in inferred.graph.output)
    excluded.update(item.name for item in inferred.graph.initializer)
    memory = 0
    seen: set[str] = set()
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in excluded or name in seen:
                continue
            seen.add(name)
            tensor = infos[name].type.tensor_type
            elements = math.prod(int(dim.dim_value) for dim in tensor.shape.dim)
            dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)
            memory += elements * np.dtype(dtype).itemsize
    params = 0
    for item in inferred.graph.initializer:
        params += math.prod(item.dims) if item.dims else 1
    return int(memory + params)


def main() -> int:
    source = json.loads(SOURCE.read_text())
    retained: list[dict[str, object]] = []
    reject_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        reject_counts[reason] = reject_counts.get(reason, 0) + 1

    for record in source["candidates"]:
        task = int(record["task"])
        if task in QUARANTINED:
            reject("quarantined_task")
            continue
        candidate_path = ROOT / record["path"]
        baseline_path = BASE_DIR / f"task{task:03d}.onnx"
        try:
            candidate = onnx.load(candidate_path, load_external_data=False)
            baseline = onnx.load(baseline_path, load_external_data=False)
            if computational_payload(candidate) != computational_payload(baseline):
                reject("computational_payload_differs")
                continue
            onnx.checker.check_model(candidate, full_check=True)
            shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
            ub = bias_ub(candidate)
            if ub:
                reject("conv_bias_ub")
                continue
            base_vi = {value.name: dims(value) for value in baseline.graph.value_info}
            cand_vi = {value.name: dims(value) for value in candidate.graph.value_info}
            differences = [
                {"name": name, "baseline": base_vi.get(name), "candidate": cand_vi.get(name)}
                for name in sorted(set(base_vi) | set(cand_vi))
                if base_vi.get(name) != cand_vi.get(name)
            ]
            retained.append({
                "task": task,
                "source_path": record["path"],
                "candidate_sha256": sha256(candidate_path),
                "baseline_sha256": sha256(baseline_path),
                "value_info_differences": differences,
                "static_cost_candidate": static_cost(candidate),
                "static_cost_baseline": static_cost(baseline),
                "conv_bias_ub": ub,
                "checker_full": "PASS",
                "strict_shape_inference_data_prop": "PASS",
            })
        except Exception as exc:
            reject(f"error:{type(exc).__name__}")

    # Deduplicate archive aliases of the same candidate bytes.
    unique: dict[tuple[int, str], dict[str, object]] = {}
    for row in retained:
        unique[(int(row["task"]), str(row["candidate_sha256"]))] = row
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "scanned": len(source["candidates"]),
        "quarantined_tasks": sorted(QUARANTINED),
        "reject_counts": dict(sorted(reject_counts.items())),
        "retained": sorted(unique.values(), key=lambda row: int(row["task"])),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
