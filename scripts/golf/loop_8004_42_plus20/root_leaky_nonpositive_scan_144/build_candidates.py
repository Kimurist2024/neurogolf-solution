#!/usr/bin/env python3
"""Find Mul(nonpositive, positive scalar) -> LeakyRelu attribute shaves."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = REPO / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"

NONNEG = "nonnegative"
NONPOS = "nonpositive"
ZERO = "zero"
UNKNOWN = "unknown"


def scalar_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        init.name: np.asarray(numpy_helper.to_array(init))
        for init in model.graph.initializer
    }


def sign_of_array(value: np.ndarray) -> str:
    if value.size == 0 or not np.all(np.isfinite(value)):
        return UNKNOWN
    if np.all(value == 0):
        return ZERO
    if np.all(value >= 0):
        return NONNEG
    if np.all(value <= 0):
        return NONPOS
    return UNKNOWN


def sign_map(model: onnx.ModelProto, arrays: dict[str, np.ndarray]) -> tuple[dict[str, str], dict[str, str]]:
    signs = {name: sign_of_array(value) for name, value in arrays.items()}
    proofs = {name: "initializer values" for name in arrays}
    for item in model.graph.input:
        # NeuroGolf's sole input is one-hot/binary and therefore nonnegative.
        signs[item.name] = NONNEG
        proofs[item.name] = "benchmark one-hot graph input"

    preserve = {
        "Cast", "CastLike", "Identity", "Reshape", "Transpose", "Expand",
        "Squeeze", "Unsqueeze", "Flatten", "Tile", "Gather", "GatherElements",
        "GatherND", "Slice", "CenterCropPad", "Compress", "ScatterElements",
        "ScatterND", "ReduceSum", "ReduceMax", "ReduceMin", "GlobalAveragePool",
        "GlobalMaxPool", "AveragePool", "MaxPool",
    }
    force_nonneg = {
        "Abs", "Relu", "HardSigmoid", "Sigmoid", "Softmax", "Softplus",
        "ReduceL1", "ReduceL2", "ReduceLogSumExp", "ReduceSumSquare",
        "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "And",
        "Or", "Xor", "Not", "IsInf", "IsNaN", "Shape", "Size", "NonZero",
        "ArgMax", "ArgMin", "OneHot",
    }

    # Graphs are topologically sorted. Repeat to cover local functions or odd
    # ordering without pretending unknown signs are proved.
    for _ in range(3):
        changed = False
        for node in model.graph.node:
            if not node.output:
                continue
            out = node.output[0]
            ins = [signs.get(name, UNKNOWN) for name in node.input]
            value = UNKNOWN
            why = ""
            if node.op_type in force_nonneg:
                value, why = NONNEG, f"{node.op_type} has nonnegative range"
            elif node.op_type in preserve and ins:
                value, why = ins[0], f"{node.op_type} preserves source sign"
            elif node.op_type == "Neg" and ins:
                value = {NONNEG: NONPOS, NONPOS: NONNEG, ZERO: ZERO}.get(ins[0], UNKNOWN)
                why = "Neg flips proved source sign"
            elif node.op_type == "Add" and len(ins) >= 2:
                if all(item in {NONNEG, ZERO} for item in ins):
                    value, why = NONNEG, "sum of nonnegative inputs"
                elif all(item in {NONPOS, ZERO} for item in ins):
                    value, why = NONPOS, "sum of nonpositive inputs"
            elif node.op_type == "Sub" and len(ins) == 2:
                if ins[0] in {NONNEG, ZERO} and ins[1] in {NONPOS, ZERO}:
                    value, why = NONNEG, "nonnegative minus nonpositive"
                elif ins[0] in {NONPOS, ZERO} and ins[1] in {NONNEG, ZERO}:
                    value, why = NONPOS, "nonpositive minus nonnegative"
            elif node.op_type in {"Mul", "Div"} and len(ins) >= 2:
                if ZERO in ins:
                    value, why = ZERO, f"{node.op_type} with proved zero"
                elif all(item == NONNEG for item in ins):
                    value, why = NONNEG, f"{node.op_type} of nonnegative inputs"
                elif ins.count(NONPOS) == 1 and all(item in {NONNEG, NONPOS} for item in ins):
                    value, why = NONPOS, f"{node.op_type} has one nonpositive factor"
                elif ins.count(NONPOS) == 2:
                    value, why = NONNEG, f"{node.op_type} has two nonpositive factors"
            elif node.op_type == "PRelu" and ins:
                if ins[0] in {NONNEG, ZERO}:
                    value, why = NONNEG, "PRelu leaves nonnegative source unchanged"
                elif len(ins) > 1 and ins[0] == NONPOS and ins[1] in {NONNEG, ZERO}:
                    value, why = NONPOS, "PRelu positive slope preserves nonpositive sign"
            if value != UNKNOWN and signs.get(out) != value:
                signs[out] = value
                proofs[out] = why
                changed = True
        if not changed:
            break
    return signs, proofs


def profile(model: onnx.ModelProto, task: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"leaky144_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def remove_initializer(model: onnx.ModelProto, name: str) -> None:
    keep = [item for item in model.graph.initializer if item.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    inventory: list[dict] = []
    winners: list[dict] = []
    errors: list[dict] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".onnx")):
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            arrays = scalar_arrays(model)
            signs, proofs = sign_map(model, arrays)
            uses = Counter(name for node in model.graph.node for name in node.input)
            by_init: dict[str, list[int]] = {}
            for index, node in enumerate(model.graph.node):
                if node.op_type != "Mul" or len(node.input) != 2:
                    continue
                for scalar_pos in (0, 1):
                    scalar_name = node.input[scalar_pos]
                    dynamic_name = node.input[1 - scalar_pos]
                    value = arrays.get(scalar_name)
                    if value is None or value.size != 1:
                        continue
                    scale = float(value.reshape(-1)[0])
                    if not math.isfinite(scale) or scale == 0:
                        continue
                    if signs.get(dynamic_name) not in {NONPOS, ZERO}:
                        continue
                    row = {
                        "task": task,
                        "node_index": index,
                        "initializer": scalar_name,
                        "scale": scale,
                        "dynamic_input": dynamic_name,
                        "dynamic_sign": signs.get(dynamic_name),
                        "proof": proofs.get(dynamic_name, ""),
                        "initializer_uses": uses[scalar_name],
                    }
                    inventory.append(row)
                    by_init.setdefault(scalar_name, []).append(index)

            for init_name, indices in by_init.items():
                # The initializer must become dead, and every use must be one of
                # the proved target Mul nodes being replaced together.
                if uses[init_name] != len(indices):
                    continue
                candidate = copy.deepcopy(model)
                scale = float(arrays[init_name].reshape(-1)[0])
                try:
                    for index in indices:
                        old = candidate.graph.node[index]
                        dynamic = old.input[0] if old.input[1] == init_name else old.input[1]
                        replacement = helper.make_node(
                            "LeakyRelu", [dynamic], list(old.output),
                            name=old.name, alpha=scale,
                        )
                        candidate.graph.node[index].CopyFrom(replacement)
                    remove_initializer(candidate, init_name)
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    baseline_profile = profile(model, task)
                    candidate_profile = profile(candidate, task)
                    row = {
                        "task": task,
                        "initializer": init_name,
                        "scale": scale,
                        "node_indices": indices,
                        "baseline": baseline_profile,
                        "candidate": candidate_profile,
                        "strict_lower": candidate_profile["cost"] < baseline_profile["cost"],
                    }
                    if row["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{init_name}.onnx"
                        onnx.save(candidate, path)
                        row["path"] = str(path)
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                        winners.append(row)
                except Exception as exc:
                    errors.append({
                        "task": task,
                        "initializer": init_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
    result = {"inventory": inventory, "winners": winners, "errors": errors}
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "eligible_nodes": len(inventory),
        "strict_lower": winners,
        "errors": errors,
    }, indent=2))


if __name__ == "__main__":
    main()
