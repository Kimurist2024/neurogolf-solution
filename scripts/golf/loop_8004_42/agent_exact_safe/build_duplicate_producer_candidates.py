#!/usr/bin/env python3
"""Build exact duplicate-producer eliminations from the immutable 8004.42 base.

Only the two previously audited duplicate node pairs are considered.  A
candidate is emitted only when the producer NodeProto messages are identical
after clearing their names and output names.  This makes the rewrite a pure
common-subexpression elimination with no floating-point reassociation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "scripts/golf/loop_8004_42/submission_8004.42_fixed_rebase_meta.zip"
PAIRS = {
    165: ("__sp_low", "__sp_low2"),
    169: ("negF", "Fsh_neg"),
}


def tensor_key(tensor: onnx.TensorProto) -> bytes:
    clone = copy.deepcopy(tensor)
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_node(node: onnx.NodeProto) -> bytes:
    clone = copy.deepcopy(node)
    clone.name = ""
    del clone.output[:]
    return clone.SerializeToString(deterministic=True)


def build(task: int, keep: str, remove: str, source: bytes) -> dict[str, object]:
    model = onnx.load_model_from_string(source)
    producers = {
        output: (index, node)
        for index, node in enumerate(model.graph.node)
        for output in node.output
        if output
    }
    keep_index, keep_node = producers[keep]
    remove_index, remove_node = producers[remove]
    if canonical_node(keep_node) != canonical_node(remove_node):
        raise ValueError(f"task{task:03d}: producers are not identical")

    candidate = copy.deepcopy(model)
    del candidate.graph.node[remove_index]
    for node in candidate.graph.node:
        for index, name in enumerate(node.input):
            if name == remove:
                node.input[index] = keep
    # Remove stale annotations for the eliminated value, if present.
    kept_info = [value for value in candidate.graph.value_info if value.name != remove]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(kept_info)

    onnx.checker.check_model(candidate, full_check=True)
    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)

    baseline_path = HERE / "baseline" / f"task{task:03d}.onnx"
    candidate_path = HERE / "models" / f"task{task:03d}.onnx"
    onnx.save_model(model, baseline_path)
    onnx.save_model(candidate, candidate_path)
    candidate_bytes = candidate_path.read_bytes()
    return {
        "task": task,
        "rewrite": "exact_duplicate_producer_elimination",
        "keep_output": keep,
        "remove_output": remove,
        "keep_node_index": keep_index,
        "remove_node_index": remove_index,
        "baseline_sha256": digest(source),
        "candidate_sha256": digest(candidate_bytes),
        "baseline_nodes": len(model.graph.node),
        "candidate_nodes": len(candidate.graph.node),
        "checker": "PASS",
        "strict_shape_inference": "PASS",
        "candidate": str(candidate_path.relative_to(ROOT)),
    }


def build_initializer_alias(task: int, keep: str, remove: str, source: bytes) -> dict[str, object]:
    model = onnx.load_model_from_string(source)
    by_name = {item.name: item for item in model.graph.initializer}
    if tensor_key(by_name[keep]) != tensor_key(by_name[remove]):
        raise ValueError(f"task{task:03d}: initializers are not byte-identical")
    candidate = copy.deepcopy(model)
    for node in candidate.graph.node:
        for index, name in enumerate(node.input):
            if name == remove:
                node.input[index] = keep
    kept = [item for item in candidate.graph.initializer if item.name != remove]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept)
    onnx.checker.check_model(candidate, full_check=True)
    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
    baseline_path = HERE / "baseline" / f"task{task:03d}.onnx"
    candidate_path = HERE / "models" / f"task{task:03d}.onnx"
    onnx.save_model(model, baseline_path)
    onnx.save_model(candidate, candidate_path)
    candidate_bytes = candidate_path.read_bytes()
    before = sum(int(numpy_helper.to_array(item).size) for item in model.graph.initializer)
    after = sum(int(numpy_helper.to_array(item).size) for item in candidate.graph.initializer)
    return {
        "task": task,
        "rewrite": "exact_byte_identical_initializer_alias",
        "keep_initializer": keep,
        "remove_initializer": remove,
        "baseline_sha256": digest(source),
        "candidate_sha256": digest(candidate_bytes),
        "baseline_params": before,
        "candidate_params": after,
        "parameter_reduction": before - after,
        "checker": "PASS",
        "strict_shape_inference": "PASS",
        "candidate": str(candidate_path.relative_to(ROOT)),
    }


def main() -> None:
    rows = []
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task, (keep, remove) in PAIRS.items():
            rows.append(build(task, keep, remove, archive.read(f"task{task:03d}.onnx")))
        rows.append(
            build_initializer_alias(
                233,
                "one_i8",
                "audit_one_i16",
                archive.read("task233.onnx"),
            )
        )
    report = {
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "base_zip_sha256": digest(BASE_ZIP.read_bytes()),
        "rows": rows,
    }
    (HERE / "build_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
