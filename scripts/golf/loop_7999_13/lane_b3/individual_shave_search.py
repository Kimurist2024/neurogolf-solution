#!/usr/bin/env python3
"""Search one-dimension value_info shaves on generator-sound B3 baselines."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import score_and_verify  # noqa: E402


BASE_COSTS = {340: 1173, 365: 1381}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static_memory(model: onnx.ModelProto) -> int:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    io = {item.name for item in list(graph.input) + list(graph.output)}
    initializers = {item.name for item in graph.initializer}
    total = 0
    for item in graph.value_info:
        if item.name in io or item.name in initializers:
            continue
        tensor_type = item.type.tensor_type
        dims = [dim.dim_value for dim in tensor_type.shape.dim]
        if any(dim <= 0 for dim in dims):
            raise ValueError(f"non-static {item.name}: {dims}")
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        total += int(np.prod(dims)) * np.dtype(dtype).itemsize
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=sorted(BASE_COSTS))
    args = parser.parse_args()

    base_path = HERE / f"baseline_task{args.task:03d}.onnx"
    base = onnx.load(base_path)
    executable = copy.deepcopy(base)
    executable.graph.ClearField("value_info")
    executable_bytes = executable.SerializeToString(deterministic=True)
    rows = []
    best = None
    base_static = static_memory(base)

    for value_index, item in enumerate(base.graph.value_info):
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for dim_index, dim in enumerate(tensor_type.shape.dim):
            if dim.dim_value <= 1:
                continue
            candidate = copy.deepcopy(base)
            changed = candidate.graph.value_info[value_index]
            before = changed.type.tensor_type.shape.dim[dim_index].dim_value
            changed.type.tensor_type.shape.dim[dim_index].dim_value = 1
            record = {
                "task": args.task,
                "value_info_index": value_index,
                "value_info": item.name,
                "dim_index": dim_index,
                "before": before,
                "after": 1,
                "checker": False,
                "strict_shape": False,
                "static_memory": None,
                "score": None,
                "error": None,
            }
            try:
                onnx.checker.check_model(candidate, full_check=True)
                record["checker"] = True
                onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                record["strict_shape"] = True
                record["static_memory"] = static_memory(candidate)
                if record["static_memory"] < base_static:
                    result = score_and_verify(
                        candidate,
                        args.task,
                        str(HERE / "tmp"),
                        label=f"{args.task}_{value_index}_{dim_index}",
                        require_correct=True,
                    )
                    record["score"] = result
                    if result and result["cost"] < BASE_COSTS[args.task]:
                        path = HERE / (
                            f"candidate_task{args.task:03d}_{value_index}_{dim_index}.onnx"
                        )
                        onnx.save(candidate, path)
                        record["candidate"] = str(path.relative_to(ROOT))
                        record["sha256"] = digest(path)
                        stripped = copy.deepcopy(candidate)
                        stripped.graph.ClearField("value_info")
                        record["executable_graph_identical"] = (
                            stripped.SerializeToString(deterministic=True) == executable_bytes
                        )
                        if best is None or result["cost"] < best[0]:
                            best = (result["cost"], path, record)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(record)

    report = {
        "task": args.task,
        "baseline_cost": BASE_COSTS[args.task],
        "baseline_static_memory": base_static,
        "probes": rows,
        "winner": best[2] if best else None,
    }
    (HERE / f"individual_shave_task{args.task:03d}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if best:
        winner_path = HERE / f"winner_task{args.task:03d}.onnx"
        winner_path.write_bytes(best[1].read_bytes())
        print(f"WINNER {winner_path}: cost={best[0]}")
    else:
        print("NO_WINNER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
