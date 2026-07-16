#!/usr/bin/env python3
"""Prove that the task109 archive shave changes annotations, not computation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "scripts/golf/loop_8003_40/base_models/task109.onnx"
CANDIDATE = HERE / "candidates/task109.onnx"
OUTPUT = HERE / "task109_semantic_audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value_info: onnx.ValueInfoProto) -> list[int | str]:
    shape = value_info.type.tensor_type.shape
    result: list[int | str] = []
    for dim in shape.dim:
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


def main() -> None:
    base = onnx.load(BASE, load_external_data=False)
    candidate = onnx.load(CANDIDATE, load_external_data=False)
    onnx.checker.check_model(candidate, full_check=True)
    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)

    base_vi = {value.name: dims(value) for value in base.graph.value_info}
    candidate_vi = {value.name: dims(value) for value in candidate.graph.value_info}
    differences = [
        {"name": name, "baseline": base_vi.get(name), "candidate": candidate_vi.get(name)}
        for name in sorted(set(base_vi) | set(candidate_vi))
        if base_vi.get(name) != candidate_vi.get(name)
    ]
    nested = [
        f"{node.name}/{attribute.name}"
        for node in candidate.graph.node
        for attribute in node.attribute
        if attribute.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
    ]
    ops = [node.op_type for node in candidate.graph.node]
    einsum_inputs = max(
        (len(node.input) for node in candidate.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    payload_equal = computational_payload(base) == computational_payload(candidate)
    report = {
        "task": 109,
        "baseline_sha256": sha(BASE),
        "candidate_sha256": sha(CANDIDATE),
        "computational_payload_identical_after_clearing_value_info": payload_equal,
        "value_info_differences": differences,
        "checker_full": "PASS",
        "strict_shape_inference_data_prop": "PASS",
        "ops": sorted(set(ops)),
        "tfidf_or_lookup_op": any(op == "TfIdfVectorizer" for op in ops),
        "nested_graphs": nested,
        "max_einsum_inputs": einsum_inputs,
        "functions": len(candidate.functions),
        "decision": "PASS" if payload_equal and len(differences) == 1 else "FAIL",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
