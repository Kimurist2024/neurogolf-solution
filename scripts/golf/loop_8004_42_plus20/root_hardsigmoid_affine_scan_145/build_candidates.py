#!/usr/bin/env python3
"""Fuse affine chains over proved {0,1} tensors into HardSigmoid attributes."""

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


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}


def binary_values(model: onnx.ModelProto, values: dict[str, np.ndarray]) -> set[str]:
    binary: set[str] = set()
    for item in model.graph.input:
        binary.add(item.name)  # benchmark one-hot input is exactly 0/1
    for name, value in values.items():
        if value.size and np.all(np.logical_or(value == 0, value == 1)):
            binary.add(name)
    preserve = {
        "Cast", "CastLike", "Identity", "Reshape", "Transpose", "Expand",
        "Squeeze", "Unsqueeze", "Flatten", "Tile", "Gather", "GatherElements",
        "GatherND", "Slice", "CenterCropPad", "Compress", "Concat", "Split",
    }
    force = {
        "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "And",
        "Or", "Xor", "Not", "IsInf", "IsNaN", "OneHot",
    }
    for _ in range(3):
        before = len(binary)
        for node in model.graph.node:
            if not node.output:
                continue
            if node.op_type in force:
                binary.update(node.output)
            elif node.op_type in preserve and node.input and all(name in binary for name in node.input if name):
                binary.update(node.output)
            elif node.op_type in {"Max", "Min", "Mul"} and node.input and all(name in binary for name in node.input):
                binary.update(node.output)
        if len(binary) == before:
            break
    return binary


def scalar(values: dict[str, np.ndarray], name: str) -> float | None:
    value = values.get(name)
    if value is None or value.size != 1:
        return None
    result = float(value.reshape(-1)[0])
    return result if math.isfinite(result) else None


def profile(model: onnx.ModelProto, task: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"hs145_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def drop_initializers(model: onnx.ModelProto, names: set[str]) -> None:
    keep = [item for item in model.graph.initializer if item.name not in names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)


def plans(model: onnx.ModelProto) -> list[dict]:
    vals = arrays(model)
    binary = binary_values(model, vals)
    producer = {out: (idx, node) for idx, node in enumerate(model.graph.node) for out in node.output}
    uses = Counter(name for node in model.graph.node for name in node.input)
    result: list[dict] = []
    for add_idx, add in enumerate(model.graph.node):
        if add.op_type != "Add" or len(add.input) != 2:
            continue
        for affine_pos in (0, 1):
            affine_name = add.input[affine_pos]
            beta_name = add.input[1 - affine_pos]
            beta = scalar(vals, beta_name)
            source = producer.get(affine_name)
            if beta is None or source is None:
                continue
            affine_idx, affine = source
            if affine.op_type not in {"Mul", "Div"} or len(affine.input) != 2:
                continue
            if uses[affine_name] != 1:
                continue
            alpha = None
            alpha_name = ""
            dynamic = ""
            if affine.op_type == "Mul":
                for scalar_pos in (0, 1):
                    found = scalar(vals, affine.input[scalar_pos])
                    if found is not None:
                        alpha = found
                        alpha_name = affine.input[scalar_pos]
                        dynamic = affine.input[1 - scalar_pos]
                        break
            else:
                denominator = scalar(vals, affine.input[1])
                if denominator not in (None, 0.0):
                    alpha = 1.0 / denominator
                    alpha_name = affine.input[1]
                    dynamic = affine.input[0]
            if alpha is None or dynamic not in binary:
                continue
            # HardSigmoid clipping must be inactive on the complete {0,1}
            # support; otherwise the proposed affine identity is invalid.
            y0, y1 = beta, alpha + beta
            if not (0.0 <= y0 <= 1.0 and 0.0 <= y1 <= 1.0):
                continue
            removed = {alpha_name, beta_name}
            if any(uses[name] != sum(name in node.input for node in (affine, add)) for name in removed):
                continue
            result.append({
                "affine_index": affine_idx,
                "add_index": add_idx,
                "source": dynamic,
                "alpha": alpha,
                "beta": beta,
                "initializers": sorted(removed),
                "affine_op": affine.op_type,
            })
    return result


def build(model: onnx.ModelProto, plan: dict) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    add = candidate.graph.node[plan["add_index"]]
    fused = helper.make_node(
        "HardSigmoid", [plan["source"]], list(add.output), name=add.name,
        alpha=float(plan["alpha"]), beta=float(plan["beta"]),
    )
    new_nodes = []
    for idx, node in enumerate(candidate.graph.node):
        if idx == plan["affine_index"]:
            continue
        if idx == plan["add_index"]:
            new_nodes.append(fused)
        else:
            new_nodes.append(node)
    del candidate.graph.node[:]
    candidate.graph.node.extend(new_nodes)
    drop_initializers(candidate, set(plan["initializers"]))
    return candidate


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    inventory: list[dict] = []
    winners: list[dict] = []
    errors: list[dict] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".onnx")):
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            for ordinal, plan in enumerate(plans(model), 1):
                inventory.append({"task": task, **plan})
                try:
                    candidate = build(model, plan)
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    baseline = profile(model, task)
                    current = profile(candidate, task)
                    row = {
                        "task": task, "ordinal": ordinal, **plan,
                        "baseline": baseline, "candidate": current,
                        "strict_lower": current["cost"] < baseline["cost"],
                    }
                    if row["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{ordinal:02d}.onnx"
                        onnx.save(candidate, path)
                        row["path"] = str(path)
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                        winners.append(row)
                except Exception as exc:
                    errors.append({"task": task, "ordinal": ordinal, "error": f"{type(exc).__name__}: {exc}"})
    output = {"inventory": inventory, "winners": winners, "errors": errors}
    (HERE / "build.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"eligible": len(inventory), "winners": winners, "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
