#!/usr/bin/env python3
"""Replace uniform broadcast-only initializers by equal scalar tensors.

Discovery is conservative: an initializer is rewritten only when every use is
an input of an operator whose ONNX schema accepts multidirectional broadcast,
or the type-only second input of CastLike.  Promotion is intentionally left to
the normal gold/fresh/runtime/structure gates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of


BROADCAST_OPS = {
    "Add", "Sub", "Mul", "Div", "Pow", "Mod", "Fmod",
    "And", "Or", "Xor", "Equal", "Greater", "GreaterOrEqual",
    "Less", "LessOrEqual", "BitwiseAnd", "BitwiseOr", "BitwiseXor",
    "BitShift", "Max", "Min", "Mean", "Sum", "PRelu",
}


def safe_use(node: onnx.NodeProto, position: int) -> bool:
    if node.op_type in BROADCAST_OPS:
        return True
    if node.op_type == "Where":
        return position in {0, 1, 2}
    if node.op_type == "Clip":
        return position in {1, 2}
    if node.op_type == "CastLike":
        return position == 1
    return False


def broadcast_reduction(array: np.ndarray) -> np.ndarray | None:
    """Collapse axes whose slices are identical while retaining tensor rank."""
    if array.size <= 1:
        return None
    result = array
    changed = False
    for axis in range(result.ndim):
        if result.shape[axis] <= 1:
            continue
        first = np.take(result, [0], axis=axis)
        if np.issubdtype(result.dtype, np.floating) and np.isnan(first).any():
            continue
        if np.all(result == np.repeat(first, result.shape[axis], axis=axis)):
            result = first
            changed = True
    return result.copy() if changed else None


def validate(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    if model.functions or model.graph.sparse_initializer:
        raise ValueError("functions/sparse initializers forbidden")
    banned = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
    for node in model.graph.node:
        if node.op_type in banned or "Sequence" in node.op_type:
            raise ValueError(f"banned op {node.op_type}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    costs = json.loads(args.base_costs.read_text())["costs"]
    ort.set_default_logger_severity(3)
    winners: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    with zipfile.ZipFile(args.baseline) as archive:
        for task in range(1, 401):
            original = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            uses: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
            for node in original.graph.node:
                for position, name in enumerate(node.input):
                    if name:
                        uses.setdefault(name, []).append((node, position))
            candidate = copy.deepcopy(original)
            replacements: list[dict[str, object]] = []
            for index, initializer in enumerate(original.graph.initializer):
                scalar = broadcast_reduction(numpy_helper.to_array(initializer))
                consumers = uses.get(initializer.name, [])
                if scalar is None or not consumers:
                    continue
                if not all(safe_use(node, position) for node, position in consumers):
                    continue
                before = list(initializer.dims)
                candidate.graph.initializer[index].CopyFrom(
                    numpy_helper.from_array(scalar, initializer.name)
                )
                replacements.append({
                    "name": initializer.name,
                    "shape": before,
                    "elements": int(math.prod(before) if before else 1),
                    "uses": [[node.op_type, position] for node, position in consumers],
                })
            if not replacements:
                continue
            try:
                validate(candidate)
                output = args.out_dir / f"task{task:03d}.onnx"
                onnx.save(candidate, output)
                candidate_cost = int(cost_of(str(output))[2])
                base_cost = int(costs[str(task)])
                if candidate_cost <= 0 or candidate_cost >= base_cost:
                    output.unlink(missing_ok=True)
                    continue
                item = {
                    "task": task,
                    "path": str(output),
                    "baseline_cost": base_cost,
                    "candidate_cost": candidate_cost,
                    "projected_gain": math.log(base_cost / candidate_cost),
                    "replacements": replacements,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
                winners.append(item)
                print(f"task{task:03d}: {base_cost}->{candidate_cost} constants={len(replacements)}")
            except Exception as exc:
                failures.append({"task": task, "error": repr(exc), "replacements": replacements})

    winners.sort(key=lambda item: -float(item["projected_gain"]))
    payload = {
        "baseline": str(args.baseline),
        "winners": winners,
        "projected_gain": sum(float(item["projected_gain"]) for item in winners),
        "failures": failures,
    }
    (args.out_dir / "manifest_pre_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"winners": len(winners), "gain": payload["projected_gain"], "failures": len(failures)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
