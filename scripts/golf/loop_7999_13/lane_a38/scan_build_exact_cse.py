#!/usr/bin/env python3
"""Scan the 8000.46 authority ZIP for exact initializer/node/subgraph CSE."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from scripts.golf.rank_dir import cost_of  # noqa: E402


AUTHORITY = ROOT / "submission_base_8000.46.zip"
AUTHORITY_SHA256 = "74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534"
LOOKUP_RED_FLAGS = {
    "TfIdfVectorizer",
    "CategoryMapper",
    "GatherND",
    "Scatter",
    "ScatterElements",
    "ScatterND",
    "TopK",
    "ArgMax",
    "ArgMin",
    "Hardmax",
}
NONDETERMINISTIC = {
    "Bernoulli",
    "Dropout",
    "Multinomial",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tensor_key(tensor: onnx.TensorProto) -> tuple[int, tuple[int, ...], bytes]:
    array = np.asarray(numpy_helper.to_array(tensor))
    return int(tensor.data_type), tuple(int(value) for value in tensor.dims), np.ascontiguousarray(array).tobytes()


def normalized_attribute(attribute: onnx.AttributeProto) -> bytes:
    clone = copy.deepcopy(attribute)
    if clone.HasField("t"):
        clone.t.name = ""
        clone.t.doc_string = ""
    for tensor in clone.tensors:
        tensor.name = ""
        tensor.doc_string = ""
    return clone.SerializeToString(deterministic=True)


def node_key(node: onnx.NodeProto) -> tuple[str, str, tuple[str, ...], tuple[bytes, ...]]:
    return (
        node.domain,
        node.op_type,
        tuple(node.input),
        tuple(normalized_attribute(attribute) for attribute in sorted(node.attribute, key=lambda item: item.name)),
    )


def reachable_nodes(model: onnx.ModelProto) -> set[int]:
    producer = {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output
        if output
    }
    pending = [value.name for value in model.graph.output]
    reached: set[int] = set()
    while pending:
        name = pending.pop()
        index = producer.get(name)
        if index is None or index in reached:
            continue
        reached.add(index)
        pending.extend(name for name in model.graph.node[index].input if name)
    return reached


def static_shape(value: onnx.ValueInfoProto) -> bool:
    return value.type.HasField("tensor_type") and all(
        dimension.HasField("dim_value") and dimension.dim_value > 0
        for dimension in value.type.tensor_type.shape.dim
    )


def source_exclusions(model: onnx.ModelProto) -> list[str]:
    reasons: list[str] = []
    ops = [node.op_type for node in model.graph.node]
    if "CenterCropPad" in ops:
        reasons.append("CenterCropPad_lineage")
    lookup = sorted(set(ops) & LOOKUP_RED_FLAGS)
    if lookup:
        reasons.append("lookup_lineage:" + ",".join(lookup))
    if any(node.op_type == "Einsum" and len(node.input) > 16 for node in model.graph.node):
        reasons.append("giant_Einsum_lineage")
    if check_conv_bias(model):
        reasons.append("unsafe_Conv_bias")
    if model.functions or model.graph.sparse_initializer:
        reasons.append("functions_or_sparse")
    if any(
        attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attribute in node.attribute
    ):
        reasons.append("nested_graph")
    if any(node.domain not in ("", "ai.onnx") for node in model.graph.node) or any(
        item.domain not in ("", "ai.onnx") for item in model.opset_import
    ):
        reasons.append("custom_domain")
    if any(
        item.data_location == onnx.TensorProto.EXTERNAL or item.external_data
        for item in model.graph.initializer
    ):
        reasons.append("external_initializer")
    if any(
        array.dtype.kind in "fc" and not np.isfinite(array).all()
        for array in (np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer)
    ):
        reasons.append("nonfinite_initializer")
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:  # noqa: BLE001
        reasons.append("full_checker:" + type(exc).__name__)
        return reasons
    try:
        inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    except Exception as exc:  # noqa: BLE001
        reasons.append("strict_shape:" + type(exc).__name__)
        return reasons
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    if not all(static_shape(value) for value in values):
        reasons.append("nontruthful_or_dynamic_static_shape")
    return reasons


def resolve(name: str, aliases: dict[str, str]) -> str:
    while name in aliases:
        name = aliases[name]
    return name


def build(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    candidate = copy.deepcopy(model)
    graph_outputs = {value.name for value in candidate.graph.output}
    reached = reachable_nodes(candidate)
    reachable_inputs = {
        name
        for index in reached
        for name in candidate.graph.node[index].input
        if name
    }

    groups: dict[tuple[int, tuple[int, ...], bytes], list[onnx.TensorProto]] = defaultdict(list)
    for initializer in candidate.graph.initializer:
        if initializer.name in reachable_inputs or initializer.name in graph_outputs:
            groups[tensor_key(initializer)].append(initializer)
    initializer_aliases: dict[str, str] = {}
    initializer_changes: list[dict[str, object]] = []
    for tensors in groups.values():
        if len(tensors) < 2:
            continue
        canonical = next((item for item in tensors if item.name in graph_outputs), tensors[0])
        for item in tensors:
            if item.name == canonical.name or item.name in graph_outputs:
                continue
            initializer_aliases[item.name] = canonical.name
            elements = int(np.prod(item.dims, dtype=np.int64)) if item.dims else 1
            initializer_changes.append(
                {"removed": item.name, "replacement": canonical.name, "elements": elements}
            )
    for node in candidate.graph.node:
        for input_index, name in enumerate(node.input):
            node.input[input_index] = resolve(name, initializer_aliases)
    if initializer_aliases:
        kept_initializers = [
            item for item in candidate.graph.initializer if item.name not in initializer_aliases
        ]
        del candidate.graph.initializer[:]
        candidate.graph.initializer.extend(kept_initializers)

    reached = reachable_nodes(candidate)
    canonical_nodes: dict[tuple[str, str, tuple[str, ...], tuple[bytes, ...]], str] = {}
    node_aliases: dict[str, str] = {}
    node_changes: list[dict[str, object]] = []
    kept_nodes: list[onnx.NodeProto] = []
    for index, source in enumerate(candidate.graph.node):
        node = copy.deepcopy(source)
        for input_index, name in enumerate(node.input):
            node.input[input_index] = resolve(name, node_aliases)
        eligible = (
            index in reached
            and len(node.output) == 1
            and node.output[0] not in graph_outputs
            and node.op_type not in NONDETERMINISTIC
            and all(
                attribute.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for attribute in node.attribute
            )
        )
        if eligible:
            signature = node_key(node)
            if signature in canonical_nodes:
                replacement = canonical_nodes[signature]
                node_aliases[node.output[0]] = replacement
                node_changes.append(
                    {
                        "removed": node.output[0],
                        "replacement": replacement,
                        "op": node.op_type,
                        "constant_payload_cse": node.op_type == "Constant",
                    }
                )
                continue
            canonical_nodes[signature] = node.output[0]
        kept_nodes.append(node)
    for node in kept_nodes:
        for input_index, name in enumerate(node.input):
            node.input[input_index] = resolve(name, node_aliases)
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept_nodes)
    removed_names = set(initializer_aliases) | set(node_aliases)
    kept_value_info = [value for value in candidate.graph.value_info if value.name not in removed_names]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(kept_value_info)
    onnx.checker.check_model(candidate, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(candidate), strict_mode=True, data_prop=True)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    if not all(static_shape(value) for value in values):
        raise RuntimeError("candidate lost static truthful shapes")
    return candidate, {
        "initializer_changes": initializer_changes,
        "node_changes": node_changes,
        "constant_payload_cse": sum(bool(row["constant_payload_cse"]) for row in node_changes),
        "deterministic_node_cse": sum(not bool(row["constant_payload_cse"]) for row in node_changes),
        "initializer_elements_removed": sum(int(row["elements"]) for row in initializer_changes),
    }


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"a38_{task:03d}_") as directory:
        path = Path(directory) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        return tuple(int(value) for value in cost_of(str(path)))


def main() -> None:
    authority_hash = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    if authority_hash != AUTHORITY_SHA256:
        raise RuntimeError("authority ZIP hash mismatch")
    candidates = HERE / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            payload = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_model(io.BytesIO(payload))
            exclusions = source_exclusions(model)
            if exclusions:
                excluded.append({"task": task, "reasons": exclusions})
                continue
            try:
                candidate, changes = build(model)
                if not changes["initializer_changes"] and not changes["node_changes"]:
                    continue
                baseline_memory, baseline_parameters, baseline_cost = measure(model, task)
                memory, parameters, cost = measure(candidate, task)
                row = {
                    "task": task,
                    "source_sha256": sha256_bytes(payload),
                    **changes,
                    "baseline_memory": baseline_memory,
                    "baseline_parameters": baseline_parameters,
                    "baseline_cost": baseline_cost,
                    "candidate_memory": memory,
                    "candidate_parameters": parameters,
                    "candidate_cost": cost,
                }
                if cost < baseline_cost:
                    output = candidates / f"task{task:03d}.onnx"
                    onnx.save(candidate, output)
                    row["candidate"] = str(output.relative_to(ROOT))
                    row["candidate_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
                    row["projected_gain"] = float(np.log(baseline_cost / cost))
                    rows.append(row)
                else:
                    row["reason"] = "no_actual_cost_reduction"
                    errors.append(row)
            except Exception as exc:  # noqa: BLE001
                errors.append({"task": task, "error": f"{type(exc).__name__}: {exc}"})
            if task % 50 == 0:
                print(json.dumps({"scanned": task, "candidates": len(rows), "excluded": len(excluded), "errors": len(errors)}), flush=True)
    rows.sort(key=lambda row: (int(row["task"]) < 150, -float(row["projected_gain"]), int(row["task"])))
    manifest = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "task_count": 400,
        "candidate_count": len(rows),
        "candidates": rows,
        "excluded_count": len(excluded),
        "excluded": excluded,
        "errors": errors,
        "source_zip_modified": False,
    }
    (HERE / "scan_build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "candidate_count": len(rows),
        "candidate_tasks": [row["task"] for row in rows],
        "projected_gain": sum(float(row["projected_gain"]) for row in rows),
        "excluded_count": len(excluded),
        "errors": len(errors),
    }, indent=2))


if __name__ == "__main__":
    main()
