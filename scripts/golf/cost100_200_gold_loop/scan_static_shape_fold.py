#!/usr/bin/env python3
"""Fold Shape(static_tensor) into small int64 initializers exactly."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.23.zip"
OUT = HERE / "static_shape_candidates"
EVIDENCE = HERE / "static_shape_evidence.json"

sys.path.insert(0, str(ROOT))
from scripts.golf import try_candidate  # noqa: E402
from scripts.lib import scoring  # noqa: E402


EXCLUDED = {
    9, 12, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 110,
    112, 133, 134, 138, 145, 157, 158, 161, 169, 170, 173, 174, 175, 178,
    185, 187, 188, 191, 192, 196, 198, 202, 205, 208, 209, 216, 219, 222,
    233, 246, 255, 277, 285, 286, 302, 319, 325, 333, 343, 346, 355, 361,
    365, 366, 372, 377, 379, 391, 393, 396,
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def costs() -> dict[int, int]:
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        return {
            int(row["task"][4:]): int(row["cost"])
            for row in csv.DictReader(handle)
            if 100 <= int(row["cost"]) <= 200
            and int(row["task"][4:]) not in EXCLUDED
        }


def dims(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def known_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=False, data_prop=False
    )
    result = {
        item.name: list(item.dims)
        for item in inferred.graph.initializer
    }
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    ):
        shape = dims(value)
        if shape is not None:
            result[value.name] = shape
    return result


def shape_value(node: onnx.NodeProto, input_dims: list[int]) -> np.ndarray:
    attrs = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
    rank = len(input_dims)
    start = int(attrs.get("start", 0))
    end = int(attrs.get("end", rank))
    if start < 0:
        start += rank
    if end < 0:
        end += rank
    start = max(0, min(rank, start))
    end = max(0, min(rank, end))
    return np.asarray(input_dims[start:end], dtype=np.int64)


def build(original: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, Any]]]:
    model = copy.deepcopy(original)
    shapes = known_shapes(model)
    replacements: dict[str, str] = {}
    additions = []
    removed_indices = []
    plans = []
    graph_outputs = {item.name for item in model.graph.output}
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Shape" or len(node.input) != 1 or len(node.output) != 1:
            continue
        if node.output[0] in graph_outputs or node.input[0] not in shapes:
            continue
        value = shape_value(node, shapes[node.input[0]])
        name = f"static_shape_{index}"
        additions.append(numpy_helper.from_array(value, name))
        replacements[node.output[0]] = name
        removed_indices.append(index)
        plans.append(
            {
                "node_index": index,
                "input": node.input[0],
                "input_shape": shapes[node.input[0]],
                "output": node.output[0],
                "constant_name": name,
                "constant_value": value.tolist(),
            }
        )
    if not plans:
        return model, []
    for node in model.graph.node:
        for position, name in enumerate(node.input):
            if name in replacements:
                node.input[position] = replacements[name]
    kept = [
        node for index, node in enumerate(model.graph.node) if index not in removed_indices
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.graph.initializer.extend(additions)
    kept_info = [
        value for value in model.graph.value_info if value.name not in replacements
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_info)
    return model, plans


def main() -> None:
    target_costs = costs()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "authority": AUTHORITY.name,
        "authority_sha256": sha(AUTHORITY.read_bytes()),
        "scope": sorted(target_costs),
        "rows": [],
        "winners": [],
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in sorted(target_costs):
            source = archive.read(f"task{task:03d}.onnx")
            original = onnx.load_model_from_string(source)
            candidate, plans = build(original)
            row: dict[str, Any] = {
                "task": task,
                "authority_cost": target_costs[task],
                "plans": plans,
            }
            if not plans:
                report["rows"].append(row)
                continue
            try:
                structure = try_candidate._validate_ops_and_shapes(candidate)
            except Exception as exc:  # noqa: BLE001
                structure = False
                row["structural_error"] = f"{type(exc).__name__}: {exc}"
            row["structural_ok"] = structure
            if not structure:
                report["rows"].append(row)
                continue
            try:
                with tempfile.TemporaryDirectory(
                    prefix=f"shapefold_{task:03d}_", dir="/tmp"
                ) as work:
                    scored = scoring.score_and_verify(
                        copy.deepcopy(candidate),
                        task,
                        work,
                        label="candidate",
                        require_correct=True,
                    )
            except Exception as exc:  # noqa: BLE001
                scored = None
                row["profile_error"] = f"{type(exc).__name__}: {exc}"
            row["profile"] = scored
            if scored is None or int(scored["cost"]) >= target_costs[task]:
                report["rows"].append(row)
                continue
            gold_ok, mismatch = try_candidate._verify_gold(candidate, task)
            margin_ok, minimum = try_candidate._check_margin(candidate, task)
            row.update(
                {
                    "gold_exact": gold_ok,
                    "gold_mismatch": None
                    if mismatch is None
                    else {"subset": mismatch.subset, "index": mismatch.index},
                    "margin_ok": margin_ok,
                    "minimum_positive": minimum,
                }
            )
            if not gold_ok or not margin_ok:
                report["rows"].append(row)
                continue
            data = candidate.SerializeToString()
            candidate_cost = int(scored["cost"])
            path = OUT / f"task{task:03d}_cost{candidate_cost}_static_shape.onnx"
            path.write_bytes(data)
            winner = {
                **row,
                "candidate_cost": candidate_cost,
                "score_gain": math.log(target_costs[task] / candidate_cost),
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(data),
                "proof": "Shape is applied to a statically-shaped tensor and replaced by its exact int64 shape vector",
            }
            report["rows"].append(winner)
            report["winners"].append(winner)
            print(
                json.dumps(
                    {
                        "task": task,
                        "authority_cost": target_costs[task],
                        "candidate_cost": candidate_cost,
                        "gold_exact": gold_ok,
                    }
                ),
                flush=True,
            )
    report["winner_count"] = len(report["winners"])
    report["total_gain"] = sum(item["score_gain"] for item in report["winners"])
    EVIDENCE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "winner_count": report["winner_count"],
                "total_gain": report["total_gain"],
                "evidence": str(EVIDENCE.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
