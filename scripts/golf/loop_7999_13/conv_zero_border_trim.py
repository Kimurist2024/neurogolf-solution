#!/usr/bin/env python3
"""Trim all-zero Conv/QLinearConv kernel borders with compensating pads."""

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


def attr_map(node: onnx.NodeProto) -> dict[str, onnx.AttributeProto]:
    return {attr.name: attr for attr in node.attribute}


def kernel_input(node: onnx.NodeProto) -> int | None:
    if node.op_type == "Conv":
        return 1
    if node.op_type == "QLinearConv":
        return 3
    return None


def trim_for(array: np.ndarray, pads: list[int], dilations: list[int]) -> tuple[np.ndarray, list[int], list[int], list[int]] | None:
    spatial = array.ndim - 2
    starts = [0] * spatial
    ends = list(array.shape[2:])
    for axis in range(spatial):
        full_axis = axis + 2
        while starts[axis] + 1 < ends[axis]:
            sl = [slice(None)] * array.ndim
            sl[full_axis] = starts[axis]
            if np.any(array[tuple(sl)] != 0):
                break
            starts[axis] += 1
        while ends[axis] - 1 > starts[axis]:
            sl = [slice(None)] * array.ndim
            sl[full_axis] = ends[axis] - 1
            if np.any(array[tuple(sl)] != 0):
                break
            ends[axis] -= 1
    if not any(starts) and ends == list(array.shape[2:]):
        return None
    new_pads = pads.copy()
    for axis in range(spatial):
        lead = starts[axis] * dilations[axis]
        tail = (array.shape[axis + 2] - ends[axis]) * dilations[axis]
        if new_pads[axis] < lead or new_pads[spatial + axis] < tail:
            return None
        new_pads[axis] -= lead
        new_pads[spatial + axis] -= tail
    slices = (slice(None), slice(None), *(slice(starts[i], ends[i]) for i in range(spatial)))
    return array[slices].copy(), new_pads, starts, ends


def validate(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    if model.functions or model.graph.sparse_initializer:
        raise ValueError("functions/sparse initializers forbidden")


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
            initializers = {item.name: item for item in original.graph.initializer}
            uses = {name: 0 for name in initializers}
            for node in original.graph.node:
                for name in node.input:
                    if name in uses:
                        uses[name] += 1
            candidate = copy.deepcopy(original)
            candidate_inits = {item.name: item for item in candidate.graph.initializer}
            changes: list[dict[str, object]] = []
            for node in candidate.graph.node:
                position = kernel_input(node)
                if position is None or position >= len(node.input):
                    continue
                name = node.input[position]
                if name not in candidate_inits or uses.get(name) != 1:
                    continue
                array = numpy_helper.to_array(candidate_inits[name])
                if array.ndim < 3:
                    continue
                attrs = attr_map(node)
                auto_pad = onnx.helper.get_attribute_value(attrs["auto_pad"]) if "auto_pad" in attrs else b"NOTSET"
                if auto_pad not in {b"NOTSET", "NOTSET"}:
                    continue
                spatial = array.ndim - 2
                pads = list(onnx.helper.get_attribute_value(attrs["pads"])) if "pads" in attrs else [0] * (2 * spatial)
                dilations = list(onnx.helper.get_attribute_value(attrs["dilations"])) if "dilations" in attrs else [1] * spatial
                result = trim_for(array, pads, dilations)
                if result is None:
                    continue
                trimmed, new_pads, starts, ends = result
                before = list(array.shape)
                candidate_inits[name].CopyFrom(numpy_helper.from_array(trimmed, name))
                if "pads" in attrs:
                    attrs["pads"].ints[:] = new_pads
                elif any(new_pads):
                    node.attribute.append(onnx.helper.make_attribute("pads", new_pads))
                if "kernel_shape" in attrs:
                    attrs["kernel_shape"].ints[:] = list(trimmed.shape[2:])
                changes.append({"node": node.name, "weight": name, "before": before, "after": list(trimmed.shape), "starts": starts, "ends": ends, "pads": new_pads})
            if not changes:
                continue
            try:
                validate(candidate)
                path = args.out_dir / f"task{task:03d}.onnx"
                onnx.save(candidate, path)
                candidate_cost = int(cost_of(str(path))[2])
                base_cost = int(costs[str(task)])
                if candidate_cost <= 0 or candidate_cost >= base_cost:
                    path.unlink(missing_ok=True)
                    continue
                item = {"task": task, "path": str(path), "baseline_cost": base_cost, "candidate_cost": candidate_cost, "projected_gain": math.log(base_cost / candidate_cost), "changes": changes, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                winners.append(item)
                print(f"task{task:03d}: {base_cost}->{candidate_cost} trims={len(changes)}")
            except Exception as exc:
                failures.append({"task": task, "error": repr(exc), "changes": changes})
    winners.sort(key=lambda item: -float(item["projected_gain"]))
    payload = {"baseline": str(args.baseline), "winners": winners, "projected_gain": sum(float(item["projected_gain"]) for item in winners), "failures": failures}
    (args.out_dir / "manifest_pre_validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"winners": len(winners), "gain": payload["projected_gain"], "failures": len(failures)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
