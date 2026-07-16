#!/usr/bin/env python3
"""Exact static Shape/ConstantOfShape folding for the cost-351..500 lane."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LANE = import_path("cost351_500_lane_support", HERE / "worker.py")


def static_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for item in model.graph.initializer:
        result[item.name] = tuple(int(dim) for dim in item.dims)
    for value in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        ttype = value.type.tensor_type
        if not ttype.HasField("shape"):
            continue
        dims = ttype.shape.dim
        if dims and all(dim.HasField("dim_value") and dim.dim_value >= 0 for dim in dims):
            result[value.name] = tuple(int(dim.dim_value) for dim in dims)
    try:
        inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=False)
    except Exception:
        inferred = None
    if inferred is not None:
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        ):
            ttype = value.type.tensor_type
            if not ttype.HasField("shape"):
                continue
            dims = ttype.shape.dim
            if dims and all(dim.HasField("dim_value") and dim.dim_value >= 0 for dim in dims):
                result[value.name] = tuple(int(dim.dim_value) for dim in dims)
    return result


def attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def shape_slice(shape: tuple[int, ...], node: onnx.NodeProto) -> np.ndarray:
    rank = len(shape)
    start = attr_int(node, "start", 0)
    end = attr_int(node, "end", rank)
    if start < 0:
        start += rank
    if end < 0:
        end += rank
    start = min(max(start, 0), rank)
    end = min(max(end, 0), rank)
    return np.asarray(shape[start:end], dtype=np.int64)


def constant_of_shape_value(node: onnx.NodeProto) -> np.ndarray:
    for attr in node.attribute:
        if attr.name == "value":
            return np.asarray(numpy_helper.to_array(attr.t)).reshape(-1)[0]
    return np.asarray(0.0, dtype=np.float32)


def fold(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, Any]]]:
    candidate = copy.deepcopy(model)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in candidate.graph.initializer
    }
    shapes = static_shapes(candidate)
    folded: list[dict[str, Any]] = []
    kept: list[onnx.NodeProto] = []

    for index, node in enumerate(candidate.graph.node):
        array: np.ndarray | None = None
        if node.op_type == "Shape" and len(node.input) == 1 and len(node.output) == 1:
            input_shape = shapes.get(node.input[0])
            if input_shape is not None:
                array = shape_slice(input_shape, node)
        elif (
            node.op_type == "ConstantOfShape"
            and len(node.input) == 1
            and len(node.output) == 1
            and node.input[0] in arrays
        ):
            dims = np.asarray(arrays[node.input[0]], dtype=np.int64).reshape(-1)
            if np.all(dims >= 0):
                scalar = constant_of_shape_value(node)
                array = np.full(tuple(int(x) for x in dims), scalar, dtype=scalar.dtype)

        if array is None:
            kept.append(copy.deepcopy(node))
            continue

        output = node.output[0]
        arrays[output] = array
        shapes[output] = tuple(array.shape)
        candidate.graph.initializer.append(numpy_helper.from_array(array, name=output))
        folded.append(
            {
                "node_index": index,
                "op_type": node.op_type,
                "output": output,
                "replacement_dtype": str(array.dtype),
                "replacement_shape": list(array.shape),
                "replacement_values": array.reshape(-1).tolist(),
            }
        )

    del candidate.graph.node[:]
    candidate.graph.node.extend(kept)
    return candidate, folded


def fresh_exact(data: bytes, task: int) -> list[dict[str, Any]]:
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    result: list[dict[str, Any]] = []
    for seed in (351_500_100_000 + task, 351_500_200_000 + task):
        cases, generation = LANE.BASE.SUPPORT.fresh_cases(task, seed, task_map)
        runtime = LANE.BASE.failfast_known(data, cases)
        result.append(
            {
                "seed": seed,
                "generation": generation,
                "runtime": runtime,
                "pass": bool(
                    runtime.get("early_reject_reason") is None
                    and LANE.BASE.runtime_pass(runtime)
                ),
            }
        )
    return result


def main() -> int:
    generated_dir = HERE / "static_fold_generated"
    accepted_dir = HERE / "candidates"
    generated_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(LANE.AUTHORITY) as archive:
        for task in LANE.ELIGIBLE:
            authority_cost = LANE.COSTS[task]
            authority = onnx.load_model_from_string(
                archive.read(f"task{task:03d}.onnx")
            )
            candidate, folded = fold(authority)
            if not folded:
                rows.append(
                    {
                        "task": task,
                        "authority_cost": authority_cost,
                        "status": "no_foldable_node",
                    }
                )
                continue

            data = candidate.SerializeToString()
            sha = hashlib.sha256(data).hexdigest()
            path = generated_dir / f"task{task:03d}_static_fold_{sha[:12]}.onnx"
            path.write_bytes(data)
            gate = LANE.official_gate(path, task, authority_cost)
            row: dict[str, Any] = {
                "task": task,
                "authority_cost": authority_cost,
                "candidate_sha256": sha,
                "candidate_path": str(path.relative_to(ROOT)),
                "folded": folded,
                "official_gate": gate,
                "status": "official_gate_reject",
            }
            if gate["pass"]:
                fresh = fresh_exact(data, task)
                row["fresh"] = fresh
                if all(item["pass"] for item in fresh):
                    candidate_cost = int(gate["candidate_cost"])
                    saved = accepted_dir / (
                        f"task{task:03d}_GOLD_cost{candidate_cost}_{sha[:12]}.onnx"
                    )
                    shutil.copy2(path, saved)
                    row.update(
                        {
                            "status": "admit",
                            "saved_path": str(saved.relative_to(ROOT)),
                            "score_gain": math.log(authority_cost / candidate_cost),
                        }
                    )
                else:
                    row["status"] = "fresh_reject"
            rows.append(row)
            print(
                json.dumps(
                    {
                        "task": task,
                        "folded": len(folded),
                        "candidate_cost": gate["candidate_cost"],
                        "status": row["status"],
                    }
                ),
                flush=True,
            )

    payload = {
        "authority": str(LANE.AUTHORITY.relative_to(ROOT)),
        "authority_sha256": LANE.AUTHORITY_SHA256,
        "method": "exact static Shape and ConstantOfShape initializer folding",
        "absolute_gate": "try_candidate official gold exact + structure + margin + fresh-2000x2 exact",
        "rows": rows,
        "admissions": [row for row in rows if row["status"] == "admit"],
        "total_gain": sum(
            float(row.get("score_gain", 0.0)) for row in rows
        ),
        "protected_writes": "lane only; root authority and ledgers untouched",
    }
    (HERE / "static_fold_evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "admissions": [
                    {
                        "task": row["task"],
                        "cost": row["official_gate"]["candidate_cost"],
                        "gain": row["score_gain"],
                    }
                    for row in payload["admissions"]
                ],
                "total_gain": payload["total_gain"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
