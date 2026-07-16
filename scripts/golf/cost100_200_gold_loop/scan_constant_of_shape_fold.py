#!/usr/bin/env python3
"""Fold direct ConstantOfShape(initializer) nodes and enforce official gold."""

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

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.23.zip"
OUT = HERE / "constant_of_shape_candidates"
EVIDENCE = HERE / "constant_of_shape_evidence.json"

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


def target_costs() -> dict[int, int]:
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        return {
            int(row["task"][4:]): int(row["cost"])
            for row in csv.DictReader(handle)
            if 100 <= int(row["cost"]) <= 200
            and int(row["task"][4:]) not in EXCLUDED
        }


def build(original: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict]]:
    model = copy.deepcopy(original)
    init = {
        tensor.name: np.asarray(numpy_helper.to_array(tensor))
        for tensor in model.graph.initializer
    }
    kept = []
    additions = []
    plans = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "ConstantOfShape" or node.input[0] not in init:
            kept.append(node)
            continue
        shape = tuple(int(value) for value in init[node.input[0]].reshape(-1))
        if not shape or any(value <= 0 for value in shape):
            kept.append(node)
            continue
        attrs = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        tensor = attrs.get("value")
        if tensor is None:
            fill = np.asarray(0, dtype=np.float32)
        else:
            fill = np.asarray(numpy_helper.to_array(tensor)).reshape(-1)[0]
        value = np.full(shape, fill, dtype=np.asarray(fill).dtype)
        additions.append(numpy_helper.from_array(value, node.output[0]))
        plans.append(
            {
                "node_index": index,
                "shape_input": node.input[0],
                "output": node.output[0],
                "shape": list(shape),
                "dtype": str(value.dtype),
                "fill": value.reshape(-1)[0].item(),
            }
        )
    if not plans:
        return model, []
    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.graph.initializer.extend(additions)
    used = {name for node in model.graph.node for name in node.input if name}
    initializers = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    return model, plans


def main() -> None:
    costs = target_costs()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "authority": AUTHORITY.name,
        "authority_sha256": sha(AUTHORITY.read_bytes()),
        "scope": sorted(costs),
        "rows": [],
        "winners": [],
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in sorted(costs):
            source = archive.read(f"task{task:03d}.onnx")
            candidate, plans = build(onnx.load_model_from_string(source))
            row = {"task": task, "authority_cost": costs[task], "plans": plans}
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
                    prefix=f"cosfold_{task:03d}_", dir="/tmp"
                ) as work:
                    profile = scoring.score_and_verify(
                        copy.deepcopy(candidate),
                        task,
                        work,
                        label="candidate",
                        require_correct=True,
                    )
            except Exception as exc:  # noqa: BLE001
                profile = None
                row["profile_error"] = f"{type(exc).__name__}: {exc}"
            row["profile"] = profile
            if profile is None or int(profile["cost"]) >= costs[task]:
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
            candidate_cost = int(profile["cost"])
            path = OUT / f"task{task:03d}_cost{candidate_cost}_cos_fold.onnx"
            path.write_bytes(data)
            winner = {
                **row,
                "candidate_cost": candidate_cost,
                "score_gain": math.log(costs[task] / candidate_cost),
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(data),
                "proof": "ConstantOfShape receives a constant positive shape and is replaced by the exact filled initializer",
            }
            report["rows"].append(winner)
            report["winners"].append(winner)
            print(
                json.dumps(
                    {
                        "task": task,
                        "authority_cost": costs[task],
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
