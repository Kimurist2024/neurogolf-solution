#!/usr/bin/env python3
"""Scan scalar binary operators that have exact unary ONNX equivalents."""

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
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = ROOT / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def profile(model: onnx.ModelProto, task: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"unary159_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def drop_dead(model: onnx.ModelProto) -> list[str]:
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    dropped = [item.name for item in model.graph.initializer if uses[item.name] == 0]
    keep = [item for item in model.graph.initializer if uses[item.name] > 0]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return dropped


def scalar(values: dict[str, np.ndarray], name: str) -> float | None:
    value = values.get(name)
    if value is None or value.size != 1 or value.dtype.kind not in "fiu":
        return None
    return float(value.reshape(-1)[0])


def build(model: onnx.ModelProto, index: int, values: dict[str, np.ndarray]) -> tuple[onnx.ModelProto, str] | None:
    node = model.graph.node[index]
    if len(node.input) != 2:
        return None
    left, right = scalar(values, node.input[0]), scalar(values, node.input[1])
    source = None
    target_op = None
    rewrite = None
    if node.op_type == "Div" and left == 1.0:
        source, target_op, rewrite = node.input[1], "Reciprocal", "Div(1,x)->Reciprocal(x)"
    elif node.op_type == "Sub" and left == 0.0:
        source, target_op, rewrite = node.input[1], "Neg", "Sub(0,x)->Neg(x)"
    elif node.op_type == "Mul" and left == -1.0:
        source, target_op, rewrite = node.input[1], "Neg", "Mul(-1,x)->Neg(x)"
    elif node.op_type == "Mul" and right == -1.0:
        source, target_op, rewrite = node.input[0], "Neg", "Mul(x,-1)->Neg(x)"
    elif node.op_type == "Max" and left == 0.0:
        source, target_op, rewrite = node.input[1], "Relu", "Max(0,x)->Relu(x)"
    elif node.op_type == "Max" and right == 0.0:
        source, target_op, rewrite = node.input[0], "Relu", "Max(x,0)->Relu(x)"
    if source is None or target_op is None or rewrite is None:
        return None
    candidate = copy.deepcopy(model)
    original = candidate.graph.node[index]
    replacement = helper.make_node(
        target_op, [source], list(original.output), name=original.name
    )
    candidate.graph.node[index].CopyFrom(replacement)
    return candidate, rewrite


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".onnx")):
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            values = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
            baseline = None
            for index, node in enumerate(model.graph.node):
                if node.op_type not in {"Div", "Sub", "Mul", "Max"}:
                    continue
                made = build(model, index, values)
                if made is None:
                    continue
                candidate, rewrite = made
                dropped = drop_dead(candidate)
                row = {"task": task, "node_index": index, "rewrite": rewrite, "dropped_initializers": dropped}
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    if baseline is None:
                        baseline = profile(model, task)
                    current = profile(candidate, task)
                    row.update(baseline=baseline, candidate=current, strict_lower=current["cost"] < baseline["cost"])
                    if row["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{index:04d}.onnx"
                        onnx.save(candidate, path)
                        row["path"] = str(path.relative_to(ROOT))
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    payload = {"authority": str(AUTHORITY.relative_to(ROOT)), "profiles": len(rows),
               "strict_lower_count": sum(bool(r.get("strict_lower")) for r in rows), "rows": rows}
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"profiles": len(rows), "strict_lower": [r for r in rows if r.get("strict_lower")],
                      "errors": len([r for r in rows if "error" in r])}, indent=2))


if __name__ == "__main__":
    main()
