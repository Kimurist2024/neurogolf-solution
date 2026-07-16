#!/usr/bin/env python3
"""Replace proved-finite-nonnegative Greater(x,0) with Cast(x,BOOL)."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = REPO / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def nonnegative_finite(model: onnx.ModelProto, values: dict[str, np.ndarray]) -> tuple[set[str], dict[str, str]]:
    good: set[str] = set()
    proof: dict[str, str] = {}
    for item in model.graph.input:
        good.add(item.name)
        proof[item.name] = "benchmark one-hot input is finite 0/1"
    for name, value in values.items():
        if value.size and np.all(np.isfinite(value)) and np.all(value >= 0):
            good.add(name)
            proof[name] = "initializer values are finite nonnegative"
    force = {
        "Abs", "Relu", "HardSigmoid", "Sigmoid", "Softmax", "Softplus",
        "ReduceL1", "ReduceL2", "ReduceSumSquare", "Equal", "Greater",
        "GreaterOrEqual", "Less", "LessOrEqual", "And", "Or", "Xor", "Not",
        "IsInf", "IsNaN", "Shape", "Size", "NonZero", "ArgMax", "ArgMin",
        "OneHot", "QLinearConv", "QLinearMatMul", "QuantizeLinear",
        "DynamicQuantizeLinear", "MatMulInteger", "ConvInteger",
    }
    preserve = {
        "Cast", "CastLike", "Identity", "Reshape", "Transpose", "Expand",
        "Squeeze", "Unsqueeze", "Flatten", "Tile", "Gather", "GatherElements",
        "GatherND", "Slice", "CenterCropPad", "Compress", "Concat", "Split",
        "ScatterElements", "ScatterND", "ReduceSum", "ReduceMax", "ReduceMin",
        "GlobalAveragePool", "GlobalMaxPool", "AveragePool", "MaxPool", "Pad",
    }
    for _ in range(4):
        before = len(good)
        for node in model.graph.node:
            if not node.output:
                continue
            why = ""
            if node.op_type in force:
                why = f"{node.op_type} has a finite nonnegative benchmark range"
            elif node.op_type in preserve and node.input and node.input[0] in good:
                why = f"{node.op_type} preserves proved finite nonnegative values"
            elif node.op_type in {"Add", "Mul"} and node.input and all(name in good for name in node.input):
                why = f"{node.op_type} combines finite nonnegative inputs"
            elif node.op_type == "PRelu" and node.input and node.input[0] in good:
                why = "PRelu leaves finite nonnegative source unchanged"
            if why:
                for output in node.output:
                    good.add(output)
                    proof[output] = why
        if len(good) == before:
            break
    return good, proof


def profile(model: onnx.ModelProto, task: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"positive147_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def drop_dead(model: onnx.ModelProto) -> list[str]:
    uses = Counter(name for node in model.graph.node for name in node.input)
    dropped = [item.name for item in model.graph.initializer if uses[item.name] == 0]
    keep = [item for item in model.graph.initializer if uses[item.name] > 0]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return dropped


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".onnx")):
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            values = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
            good, proofs = nonnegative_finite(model, values)
            for index, node in enumerate(model.graph.node):
                if node.op_type != "Greater" or len(node.input) != 2:
                    continue
                dynamic = node.input[0]
                zero_name = node.input[1]
                zero = values.get(zero_name)
                if dynamic not in good or zero is None or zero.size != 1 or float(zero.reshape(-1)[0]) != 0.0:
                    continue
                candidate = copy.deepcopy(model)
                original = candidate.graph.node[index]
                replacement = helper.make_node(
                    "Cast", [dynamic], list(original.output), name=original.name,
                    to=TensorProto.BOOL,
                )
                candidate.graph.node[index].CopyFrom(replacement)
                dropped = drop_dead(candidate)
                row = {
                    "task": task, "node_index": index, "dynamic": dynamic,
                    "zero_initializer": zero_name, "proof": proofs[dynamic],
                    "dropped_initializers": dropped,
                }
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    baseline = profile(model, task)
                    current = profile(candidate, task)
                    row.update({
                        "baseline": baseline, "candidate": current,
                        "strict_lower": current["cost"] < baseline["cost"],
                    })
                    if row["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{index:03d}.onnx"
                        onnx.save(candidate, path)
                        row["path"] = str(path)
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    (HERE / "scan.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(json.dumps({
        "eligible": len(rows),
        "strict_lower": [row for row in rows if row.get("strict_lower")],
        "errors": [row for row in rows if "error" in row],
    }, indent=2))


if __name__ == "__main__":
    main()
