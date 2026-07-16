#!/usr/bin/env python3
"""Build semantics-preserving task366 residual-golf candidates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import onnx
from onnx import numpy_helper
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "others/71407/task366.onnx"
CANDIDATES = HERE / "candidates"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_value_info(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [value for value in model.graph.value_info if value.name not in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)


def bypass_identity(model: onnx.ModelProto) -> None:
    matches = [
        (index, node)
        for index, node in enumerate(model.graph.node)
        if node.op_type == "Identity"
        and list(node.input) == ["safe_name_29"]
        and list(node.output) == ["safe_name_118"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one target Identity, got {len(matches)}")
    index, _ = matches[0]
    for node in model.graph.node:
        for input_index, name in enumerate(node.input):
            if name == "safe_name_118":
                node.input[input_index] = "safe_name_29"
    del model.graph.node[index]
    remove_value_info(model, {"safe_name_118"})


def save(model: onnx.ModelProto, name: str) -> dict[str, object]:
    path = CANDIDATES / name
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "nodes": len(model.graph.node),
        "params": sum(math.prod(init.dims) for init in model.graph.initializer),
    }


def narrow_identity_shape_to_int32(model: onnx.ModelProto) -> None:
    target = next(init for init in model.graph.initializer if init.name == "safe_name_29")
    value = numpy_helper.to_array(target)
    if value.dtype != np.int64 or value.shape != (1,) or value.tolist() != [16]:
        raise RuntimeError(f"unexpected safe_name_29: {value.dtype} {value.shape} {value.tolist()}")
    replacement = numpy_helper.from_array(value.astype(np.int32), name=target.name)
    target.CopyFrom(replacement)
    typed = {
        value.name: value
        for value in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    }
    for name in ("safe_name_118",):
        typed[name].type.tensor_type.elem_type = onnx.TensorProto.INT32


def synthesize_sixteen_from_existing_shape(model: onnx.ModelProto) -> None:
    """Reuse Shape(Gather[15]) + existing int64 one instead of a scalar initializer."""
    matches = [
        node
        for node in model.graph.node
        if node.op_type == "Identity"
        and list(node.input) == ["safe_name_29"]
        and list(node.output) == ["safe_name_118"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one target Identity, got {len(matches)}")
    node = matches[0]
    node.op_type = "Add"
    node.name = "exact_shape15_plus_one"
    del node.input[:]
    node.input.extend(["safe_name_108", "safe_name_8"])
    kept = [init for init in model.graph.initializer if init.name != "safe_name_29"]
    if len(kept) + 1 != len(model.graph.initializer):
        raise RuntimeError("safe_name_29 initializer missing or duplicated")
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def bypass_round_before_integer_casts(model: onnx.ModelProto) -> None:
    """Bypass the 21 Round nodes; support exactness is audited separately."""
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    removed_outputs: set[str] = set()
    kept: list[onnx.NodeProto] = []
    count = 0
    for node in model.graph.node:
        if node.op_type != "Round":
            kept.append(node)
            continue
        output = node.output[0]
        children = consumers.get(output, [])
        if len(children) != 1 or children[0].op_type != "Cast":
            raise RuntimeError(f"unexpected Round consumers for {node.name}: {[c.op_type for c in children]}")
        child = children[0]
        child.input[0] = node.input[0]
        removed_outputs.add(output)
        count += 1
    if count != 21:
        raise RuntimeError(f"expected 21 Round nodes, got {count}")
    del model.graph.node[:]
    model.graph.node.extend(kept)
    remove_value_info(model, removed_outputs)


SAFE_ROUNDS = {
    "safe_name_144",
    "safe_name_154",
    "safe_name_186",
    "safe_name_205",
    "safe_name_238",
    "safe_name_248",
    "safe_name_281",
    "safe_name_300",
    "safe_name_333",
    "safe_name_365",
    "safe_name_384",
    "safe_name_461",
    "safe_name_480",
    "safe_name_498",
    "safe_name_517",
    "safe_name_536",
}


def floor_divisor_and_bypass_safe_rounds(model: onnx.ModelProto) -> None:
    """Use a lower f16 log divisor at proven lowbit sites and drop Round."""
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(0.69287109375, dtype=np.float16), name="exact_lowbit_divisor")
    )
    by_output = {name: node for node in model.graph.node for name in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    removed: set[str] = set()
    kept: list[onnx.NodeProto] = []
    seen: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Round" or node.output[0] not in SAFE_ROUNDS:
            kept.append(node)
            continue
        output = node.output[0]
        div = by_output[node.input[0]]
        children = consumers.get(output, [])
        if div.op_type != "Div" or div.input[1] != "safe_name_6":
            raise RuntimeError(f"unexpected Div parent for {node.name}")
        if len(children) != 1 or children[0].op_type != "Cast":
            raise RuntimeError(f"unexpected Round consumer for {node.name}")
        div.input[1] = "exact_lowbit_divisor"
        children[0].input[0] = node.input[0]
        removed.add(output)
        seen.add(output)
    if seen != SAFE_ROUNDS:
        raise RuntimeError(f"safe Round mismatch: missing={SAFE_ROUNDS - seen} extra={seen - SAFE_ROUNDS}")
    del model.graph.node[:]
    model.graph.node.extend(kept)
    remove_value_info(model, removed)


def reuse_positive_one_for_minus_one(model: onnx.ModelProto) -> None:
    targets = []
    for node in model.graph.node:
        if node.op_type == "Add" and len(node.input) == 2 and node.input[1] == "safe_name_32":
            node.op_type = "Sub"
            node.input[1] = "safe_name_20"
            targets.append(node.name)
    if sorted(targets) != ["safe_name_755", "safe_name_757"]:
        raise RuntimeError(f"unexpected minus-one Add targets: {targets}")
    kept = [init for init in model.graph.initializer if init.name != "safe_name_32"]
    if len(kept) + 1 != len(model.graph.initializer):
        raise RuntimeError("safe_name_32 initializer missing or duplicated")
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def clip_first_window_indices(model: onnx.ModelProto) -> None:
    """Fix inherited idx=16 fresh errors using the already-live [0,15] bounds."""
    target_index = None
    for index, node in enumerate(model.graph.node):
        if node.op_type == "Gather" and node.name == "safe_name_158":
            if list(node.input) != ["safe_name_128", "safe_name_157"]:
                raise RuntimeError(f"unexpected safe_name_158 inputs: {list(node.input)}")
            target_index = index
            node.input[1] = "safe_name_157_clipped"
            break
    if target_index is None:
        raise RuntimeError("safe_name_158 Gather missing")
    clip = onnx.helper.make_node(
        "Clip",
        ["safe_name_157", "safe_name_2", "safe_name_11"],
        ["safe_name_157_clipped"],
        name="exact_no_oob_safe_name_157",
    )
    model.graph.node.insert(target_index, clip)
    model.graph.value_info.append(
        onnx.helper.make_tensor_value_info("safe_name_157_clipped", onnx.TensorProto.INT32, [8])
    )


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    baseline = onnx.load(BASE)
    shape_plus_one = copy.deepcopy(baseline)
    synthesize_sixteen_from_existing_shape(shape_plus_one)
    no_round = copy.deepcopy(baseline)
    bypass_round_before_integer_casts(no_round)
    combined = copy.deepcopy(no_round)
    synthesize_sixteen_from_existing_shape(combined)
    lowbit = copy.deepcopy(baseline)
    floor_divisor_and_bypass_safe_rounds(lowbit)
    lowbit_combined = copy.deepcopy(lowbit)
    synthesize_sixteen_from_existing_shape(lowbit_combined)
    no_error = copy.deepcopy(lowbit_combined)
    reuse_positive_one_for_minus_one(no_error)
    clip_first_window_indices(no_error)
    payload = {
        "baseline": {
            "path": str(BASE.relative_to(ROOT)),
            "sha256": digest(BASE),
            "nodes": len(baseline.graph.node),
        },
        "candidates": [
            save(shape_plus_one, "task366_shape15_plus1.onnx"),
            save(no_round, "task366_no_round.onnx"),
            save(combined, "task366_no_round_shape15_plus1.onnx"),
            save(lowbit, "task366_lowbit_floor_divisor.onnx"),
            save(lowbit_combined, "task366_lowbit_floor_divisor_shape15_plus1.onnx"),
            save(no_error, "task366_lowbit_no_oob.onnx"),
        ],
    }
    (HERE / "build.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
