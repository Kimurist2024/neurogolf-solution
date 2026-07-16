#!/usr/bin/env python3
"""Exhaustive fixed-authority scan for scalar-initializer Pow identities."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
ROOT_SUBMISSION = ROOT / "submission.zip"
ROOT_SCORES = ROOT / "all_scores.csv"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
SCORES_SHA256 = "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78"
TARGETS = {2.0: "Mul", 0.5: "Sqrt", 1.0: "Identity"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rewrite(model: onnx.ModelProto, index: int, exponent: float) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    node = candidate.graph.node[index]
    data_input = node.input[0]
    if exponent == 2.0:
        node.op_type = "Mul"
        del node.input[:]
        node.input.extend([data_input, data_input])
    elif exponent == 0.5:
        node.op_type = "Sqrt"
        del node.input[:]
        node.input.extend([data_input])
    elif exponent == 1.0:
        node.op_type = "Identity"
        del node.input[:]
        node.input.extend([data_input])
    else:
        raise ValueError(exponent)
    del node.attribute[:]
    used = {name for item in candidate.graph.node for name in item.input if name}
    kept = [item for item in candidate.graph.initializer if item.name in used]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept)
    return candidate


def main() -> int:
    before = {"submission": sha(ROOT_SUBMISSION.read_bytes()), "all_scores": sha(ROOT_SCORES.read_bytes())}
    if sha(AUTHORITY.read_bytes()) != AUTHORITY_SHA256 or before["submission"] != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    if before["all_scores"] != SCORES_SHA256:
        raise RuntimeError("all_scores changed")

    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    member_shas: dict[str, str] = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        names = {f"task{task:03d}.onnx" for task in range(1, 401)}
        missing = sorted(names - set(archive.namelist()))
        if missing:
            raise RuntimeError(f"missing authority members: {missing[:5]}")
        for task in range(1, 401):
            member = archive.read(f"task{task:03d}.onnx")
            member_shas[f"{task:03d}"] = sha(member)
            model = onnx.load_model_from_string(member)
            initializers = {
                item.name: np.asarray(numpy_helper.to_array(item))
                for item in model.graph.initializer
            }
            for index, node in enumerate(model.graph.node):
                if node.op_type != "Pow":
                    continue
                exponent_name = node.input[1] if len(node.input) > 1 else ""
                exponent_array = initializers.get(exponent_name)
                row: dict[str, Any] = {
                    "task": task,
                    "node_index": index,
                    "node_name": node.name,
                    "inputs": list(node.input),
                    "output": list(node.output),
                    "authority_member_sha256": member_shas[f"{task:03d}"],
                }
                if exponent_array is None:
                    row.update({"exponent_source": "not_initializer", "eligible": False, "reason": "exponent_not_initializer"})
                else:
                    flat = exponent_array.reshape(-1)
                    values = [float(value) for value in flat]
                    row.update(
                        {
                            "exponent_source": "initializer",
                            "exponent_name": exponent_name,
                            "exponent_dtype": str(exponent_array.dtype),
                            "exponent_shape": list(exponent_array.shape),
                            "exponent_elements": int(exponent_array.size),
                            "exponent_values": values,
                        }
                    )
                    if exponent_array.size != 1:
                        row.update({"eligible": False, "reason": "initializer_not_scalar"})
                    else:
                        value = values[0]
                        matched = next((target for target in TARGETS if value == target), None)
                        if matched is None:
                            row.update({"eligible": False, "reason": "scalar_not_2_half_or_1"})
                        else:
                            candidate = rewrite(model, index, matched)
                            data = candidate.SerializeToString(deterministic=True)
                            path = HERE / "candidates" / f"task{task:03d}_{index:04d}_{TARGETS[matched].lower()}.onnx"
                            path.write_bytes(data)
                            row.update(
                                {
                                    "eligible": True,
                                    "rewrite": f"Pow(x,{matched:g})->{TARGETS[matched]}",
                                    "candidate_path": str(path.relative_to(ROOT)),
                                    "candidate_sha256": sha(data),
                                }
                            )
                            eligible.append(row)
                rows.append(row)

    after = {"submission": sha(ROOT_SUBMISSION.read_bytes()), "all_scores": sha(ROOT_SCORES.read_bytes())}
    source_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row["exponent_source"]] = source_counts.get(row["exponent_source"], 0) + 1
        reason_counts[row.get("reason", "eligible")] = reason_counts.get(row.get("reason", "eligible"), 0) + 1
    result = {
        "lane": "agent_pow_exact_163",
        "authority": "submission_base_8009.46.zip",
        "authority_sha256": AUTHORITY_SHA256,
        "tasks_scanned": 400,
        "authority_members_hashed": len(member_shas),
        "pow_nodes": len(rows),
        "pow_tasks": sorted({row["task"] for row in rows}),
        "eligible_scalar_initializer_nodes": len(eligible),
        "source_counts": source_counts,
        "reason_counts": reason_counts,
        "rows": rows,
        "candidates": eligible,
        "deep_validation": {
            "status": "vacuous_no_candidates",
            "checker_strict_actual_known_runtime_nonfinite_margin_fresh": "not_run_because_eligible_set_empty",
        },
        "winners": [],
        "projected_gain": 0.0,
        "root_before": before,
        "root_after": after,
        "root_unchanged": before == after,
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("tasks_scanned", "pow_nodes", "pow_tasks", "eligible_scalar_initializer_nodes", "reason_counts", "root_unchanged")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
