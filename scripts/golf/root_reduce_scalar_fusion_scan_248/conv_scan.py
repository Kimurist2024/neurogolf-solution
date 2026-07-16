#!/usr/bin/env python3
"""All-400 exact Pad/shape-op absorption scan for Conv-family nodes."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
CANDIDATES = HERE / "conv_candidates"
CONVS = {"Conv", "ConvInteger", "QLinearConv"}
SHAPE_OPS = {"Squeeze", "Unsqueeze", "Transpose"}

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def typed_values(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    return {
        value.name: value
        for value in list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    }


def static_shape(value: onnx.ValueInfoProto | None) -> list[int] | None:
    if value is None:
        return None
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value < 0:
            return None
        result.append(int(dim.dim_value))
    return result


def constants(model: onnx.ModelProto) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    producers = {item.name: -1 for item in model.graph.initializer}
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                arrays[node.output[0]] = np.asarray(numpy_helper.to_array(attr.t))
                producers[node.output[0]] = node_index
            elif attr.name == "value_int":
                arrays[node.output[0]] = np.asarray(attr.i, dtype=np.int64)
                producers[node.output[0]] = node_index
            elif attr.name == "value_ints":
                arrays[node.output[0]] = np.asarray(attr.ints, dtype=np.int64)
                producers[node.output[0]] = node_index
            elif attr.name == "value_float":
                arrays[node.output[0]] = np.asarray(attr.f, dtype=np.float32)
                producers[node.output[0]] = node_index
    return arrays, producers


def uses(model: onnx.ModelProto) -> dict[str, list[tuple[int, int, onnx.NodeProto]]]:
    result: dict[str, list[tuple[int, int, onnx.NodeProto]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            result[name].append((node_index, input_index, node))
    return result


def scalar_value(arrays: dict[str, np.ndarray], name: str | None, default: int | float) -> Any | None:
    if not name:
        return default
    value = arrays.get(name)
    if value is None or value.size != 1:
        return None
    return value.reshape(-1)[0].item()


def pad_spec(
    model: onnx.ModelProto,
    pad: onnx.NodeProto,
    conv: onnx.NodeProto,
    inferred: onnx.ModelProto | None,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[int] | None]:
    row: dict[str, Any] = {}
    mode = next((attr.s.decode() for attr in pad.attribute if attr.name == "mode"), "constant")
    row["mode"] = mode
    if mode != "constant":
        row["reason"] = "Pad mode is not constant"
        return row, None
    typed = typed_values(inferred) if inferred is not None else {}
    input_shape = static_shape(typed.get(pad.input[0]))
    output_shape = static_shape(typed.get(pad.output[0]))
    row["pad_input_shape"] = input_shape
    row["pad_output_shape"] = output_shape
    if input_shape is None:
        row["reason"] = "Pad input rank/shape is not strict-static"
        return row, None
    rank = len(input_shape)
    pads_array = arrays.get(pad.input[1]) if len(pad.input) > 1 else None
    if pads_array is None:
        row["reason"] = "pads are dynamic"
        return row, None
    axes_array = arrays.get(pad.input[3]) if len(pad.input) > 3 and pad.input[3] else None
    if len(pad.input) > 3 and pad.input[3] and axes_array is None:
        row["reason"] = "Pad axes are dynamic"
        return row, None
    axes = (
        [int(value) for value in axes_array.reshape(-1)]
        if axes_array is not None
        else list(range(rank))
    )
    axes = [axis + rank if axis < 0 else axis for axis in axes]
    flat_pads = [int(value) for value in pads_array.reshape(-1)]
    row["pads"] = flat_pads
    row["axes"] = axes
    if len(flat_pads) != 2 * len(axes) or any(axis < 0 or axis >= rank for axis in axes):
        row["reason"] = "invalid static pads/axes lengths"
        return row, None
    full = [0] * (2 * rank)
    for offset, axis in enumerate(axes):
        full[axis] = flat_pads[offset]
        full[rank + axis] = flat_pads[len(axes) + offset]
    row["full_rank_pads"] = full
    if any(value < 0 for value in full):
        row["reason"] = "negative Pad cropping cannot be represented by Conv pads"
        return row, None
    if any(full[axis] or full[rank + axis] for axis in (0, 1)):
        row["reason"] = "batch/channel padding cannot be absorbed into Conv spatial pads"
        return row, None
    pad_value = scalar_value(arrays, pad.input[2] if len(pad.input) > 2 else None, 0)
    row["pad_value"] = pad_value
    if pad_value is None:
        row["reason"] = "Pad value is dynamic or non-scalar"
        return row, None
    if conv.op_type == "Conv":
        required = 0
    elif conv.op_type == "ConvInteger":
        required = scalar_value(arrays, conv.input[2] if len(conv.input) > 2 else None, 0)
    else:
        required = scalar_value(arrays, conv.input[2] if len(conv.input) > 2 else None, 0)
    row["required_implicit_pad_value"] = required
    if required is None or pad_value != required:
        row["reason"] = "Pad value does not equal Conv-family implicit input zero point"
        return row, None
    auto_pad = next((attr.s.decode() for attr in conv.attribute if attr.name == "auto_pad"), "NOTSET")
    if auto_pad not in {"", "NOTSET"}:
        row["reason"] = "Conv auto_pad is active"
        return row, None
    spatial = rank - 2
    existing = next((list(attr.ints) for attr in conv.attribute if attr.name == "pads"), [0] * (2 * spatial))
    if len(existing) != 2 * spatial:
        row["reason"] = "Conv pads attribute length is inconsistent"
        return row, None
    absorbed = [
        int(existing[index]) + full[2 + index]
        for index in range(spatial)
    ] + [
        int(existing[spatial + index]) + full[rank + 2 + index]
        for index in range(spatial)
    ]
    row["existing_conv_pads"] = existing
    row["absorbed_conv_pads"] = absorbed
    row["reason"] = None
    return row, absorbed


def build_pad_candidate(
    model: onnx.ModelProto,
    pad_index: int,
    conv_index: int,
    absorbed: list[int],
) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    pad = candidate.graph.node[pad_index]
    old_output = pad.output[0]
    source = pad.input[0]
    nodes = []
    for node_index, node in enumerate(candidate.graph.node):
        if node_index == pad_index:
            continue
        if node_index == conv_index:
            node.input[0] = source
            attrs = [attr for attr in node.attribute if attr.name != "pads"]
            del node.attribute[:]
            node.attribute.extend(attrs)
            node.attribute.append(onnx.helper.make_attribute("pads", absorbed))
        nodes.append(node)
    del candidate.graph.node[:]
    candidate.graph.node.extend(nodes)
    values = [value for value in candidate.graph.value_info if value.name != old_output]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(values)
    return candidate


def shape_transform_array(
    node: onnx.NodeProto,
    source: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    detail: dict[str, Any] = {"source_shape": list(source.shape)}
    if node.op_type == "Transpose":
        perm = next((list(attr.ints) for attr in node.attribute if attr.name == "perm"), list(reversed(range(source.ndim))))
        detail["perm"] = perm
        return np.transpose(source, axes=perm), detail
    axes = None
    if len(node.input) > 1 and node.input[1]:
        value = arrays.get(node.input[1])
        if value is not None:
            axes = tuple(int(axis) for axis in value.reshape(-1))
    if axes is None:
        for attr in node.attribute:
            if attr.name == "axes":
                axes = tuple(int(axis) for axis in attr.ints)
    detail["axes"] = list(axes) if axes is not None else None
    if axes is None:
        detail["reason"] = "axes are dynamic"
        return None, detail
    try:
        if node.op_type == "Squeeze":
            return np.squeeze(source, axis=axes), detail
        return np.expand_dims(source, axis=axes), detail
    except Exception as exc:
        detail["reason"] = f"static transform failed: {type(exc).__name__}: {exc}"
        return None, detail


def build_static_weight_candidate(
    model: onnx.ModelProto,
    transform_index: int,
    conv_index: int,
    transformed: np.ndarray,
) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    transform = candidate.graph.node[transform_index]
    source = transform.input[0]
    output = transform.output[0]
    replacement = numpy_helper.from_array(transformed, output)
    # Keep the original source only if another node uses it.
    source_other_use = any(
        source in node.input
        for index, node in enumerate(candidate.graph.node)
        if index != transform_index
    )
    initializers = [item for item in candidate.graph.initializer if item.name != output]
    if not source_other_use:
        initializers = [item for item in initializers if item.name != source]
    initializers.append(replacement)
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(initializers)
    nodes = []
    for node_index, node in enumerate(candidate.graph.node):
        if node_index == transform_index:
            continue
        if node_index == conv_index:
            for input_index, name in enumerate(node.input):
                if name == output:
                    node.input[input_index] = output
        nodes.append(node)
    del candidate.graph.node[:]
    candidate.graph.node.extend(nodes)
    return candidate


def profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"conv248_{task:03d}_{label}_") as work:
        path = Path(work) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def validate_candidate(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(copy.deepcopy(model), full_check=True)
    onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    census = Counter()
    pad_rows = []
    shape_rows = []
    errors = []
    strict_failures = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        census["authority_models"] = len(members)
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_from_string(archive.read(member))
            try:
                inferred = onnx.shape_inference.infer_shapes(
                    copy.deepcopy(model), strict_mode=True, data_prop=True
                )
            except Exception as exc:
                inferred = None
                strict_failures.append({"task": task, "error": f"{type(exc).__name__}: {exc}"})
            arrays, array_producers = constants(model)
            consumer_map = uses(model)
            baseline = None
            for node_index, node in enumerate(model.graph.node):
                if node.op_type not in {"Pad", *SHAPE_OPS} or not node.output:
                    continue
                downstream = consumer_map.get(node.output[0], [])
                if len(downstream) != 1 or downstream[0][2].op_type not in CONVS:
                    continue
                conv_index, slot, conv = downstream[0]
                if node.op_type == "Pad":
                    census["pad_single_use_to_conv_family"] += 1
                    detail, absorbed = pad_spec(model, node, conv, inferred, arrays)
                    row = {
                        "task": task,
                        "pad_node_index": node_index,
                        "conv_node_index": conv_index,
                        "conv_op": conv.op_type,
                        "conv_input_slot": slot,
                        **detail,
                    }
                    if absorbed is not None:
                        census["pad_statically_absorbable"] += 1
                        try:
                            candidate = build_pad_candidate(model, node_index, conv_index, absorbed)
                            validate_candidate(candidate)
                            baseline = baseline or profile(model, task, "authority")
                            current = profile(candidate, task, "pad")
                            row.update(
                                {
                                    "baseline": baseline,
                                    "candidate": current,
                                    "full_check": True,
                                    "strict_shape_inference_data_prop": True,
                                    "strict_lower": current["cost"] < baseline["cost"],
                                    "memory_nonincrease": current["memory"] <= baseline["memory"],
                                }
                            )
                            if row["strict_lower"] and row["memory_nonincrease"]:
                                path = CANDIDATES / f"task{task:03d}_pad_conv_absorb.onnx"
                                onnx.save(candidate, path)
                                row["path"] = str(path.relative_to(ROOT))
                                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                                census["pad_strict_lower"] += 1
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            errors.append(copy.deepcopy(row))
                    pad_rows.append(row)
                    continue

                census["shape_single_use_to_conv_family"] += 1
                row = {
                    "task": task,
                    "transform_node_index": node_index,
                    "transform_op": node.op_type,
                    "conv_node_index": conv_index,
                    "conv_op": conv.op_type,
                    "conv_input_slot": slot,
                    "source": node.input[0] if node.input else None,
                    "output": node.output[0],
                }
                weight_slot = 3 if conv.op_type == "QLinearConv" else 1
                source_array = arrays.get(node.input[0]) if node.input else None
                if slot != weight_slot:
                    row["reason"] = "transform does not feed the Conv-family weight slot"
                elif source_array is None:
                    row["reason"] = "weight transform source is dynamic"
                else:
                    transformed, detail = shape_transform_array(node, source_array, arrays)
                    row.update(detail)
                    expected = static_shape(typed_values(inferred).get(node.output[0])) if inferred is not None else None
                    row["inferred_output_shape"] = expected
                    if transformed is None:
                        row["reason"] = detail.get("reason", "static transform unavailable")
                    elif expected is not None and list(transformed.shape) != expected:
                        row["reason"] = "offline transformed weight shape disagrees with strict inference"
                    else:
                        row["reason"] = None
                        row["transformed_shape"] = list(transformed.shape)
                        census["shape_static_weight_absorbable"] += 1
                        try:
                            candidate = build_static_weight_candidate(
                                model, node_index, conv_index, transformed
                            )
                            validate_candidate(candidate)
                            baseline = baseline or profile(model, task, "authority")
                            current = profile(candidate, task, "weight_shape")
                            row.update(
                                {
                                    "baseline": baseline,
                                    "candidate": current,
                                    "full_check": True,
                                    "strict_shape_inference_data_prop": True,
                                    "strict_lower": current["cost"] < baseline["cost"],
                                    "memory_nonincrease": current["memory"] <= baseline["memory"],
                                }
                            )
                            if row["strict_lower"] and row["memory_nonincrease"]:
                                path = CANDIDATES / f"task{task:03d}_weight_shape_absorb.onnx"
                                onnx.save(candidate, path)
                                row["path"] = str(path.relative_to(ROOT))
                                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                                census["shape_strict_lower"] += 1
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            errors.append(copy.deepcopy(row))
                shape_rows.append(row)

    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
        "method": {
            "pad_absorption": "static constant Pad, spatial nonnegative pads, implicit Conv pad value/zero-point equality",
            "shape_absorption": "single-use Squeeze/Unsqueeze/Transpose with offline-static weight source only",
            "input_output_dtype_changes": False,
            "shape_cloak": False,
        },
        "census": dict(census),
        "strict_inference_authority_failures": strict_failures,
        "pad_rows": pad_rows,
        "shape_rows": shape_rows,
        "errors": errors,
        "strict_lower": [
            row for row in [*pad_rows, *shape_rows] if row.get("strict_lower")
        ],
    }
    (HERE / "conv_scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"census": dict(census), "pad_rows": pad_rows, "shape_rows": shape_rows, "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
