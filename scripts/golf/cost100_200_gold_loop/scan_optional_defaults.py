#!/usr/bin/env python3
"""Find exact parameter shaves by omitting optional ONNX default inputs.

Only rewrites whose replacement is fixed by the ONNX operator schema are
generated.  Every resulting model is still gated by the repository's official
known-data scorer with ``require_correct=True`` and the margin checker.
"""

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
OUT = HERE / "optional_default_candidates"
EVIDENCE = HERE / "optional_default_evidence.json"

sys.path.insert(0, str(ROOT))
from scripts.golf import try_candidate  # noqa: E402
from scripts.lib import scoring  # noqa: E402


# Maintained private-zero/known-black set plus the six local-gold failures from
# the rejected 8013.52 projection.
EXCLUDED = {
    9, 12, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 110,
    112, 133, 134, 138, 145, 157, 158, 161, 169, 170, 173, 174, 175, 178,
    185, 187, 188, 191, 192, 196, 198, 202, 205, 208, 209, 216, 219, 222,
    233, 246, 255, 277, 285, 286, 302, 319, 325, 333, 343, 346, 355, 361,
    365, 366, 372, 377, 379, 391, 393, 396,
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def target_costs() -> dict[int, int]:
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        return {
            int(row["task"].removeprefix("task")): int(row["cost"])
            for row in csv.DictReader(handle)
            if 100 <= int(row["cost"]) <= 200
            and int(row["task"].removeprefix("task")) not in EXCLUDED
        }


def attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }


def remove_unused_initializers(model: onnx.ModelProto) -> list[str]:
    used = {name for node in model.graph.node for name in node.input if name}
    removed = [item.name for item in model.graph.initializer if item.name not in used]
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def rewrite_once(
    original: onnx.ModelProto,
    node_index: int,
    input_index: int,
    reason: str,
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(original)
    node = model.graph.node[node_index]
    old_name = node.input[input_index]
    node.input[input_index] = ""
    removed = remove_unused_initializers(model)
    return model, {
        "node_index": node_index,
        "op": node.op_type,
        "input_index": input_index,
        "old_input": old_name,
        "removed_initializers": removed,
        "reason": reason,
    }


def opportunities(model: onnx.ModelProto) -> list[tuple[int, int, str]]:
    init = arrays(model)
    result: list[tuple[int, int, str]] = []
    for index, node in enumerate(model.graph.node):
        attr = attributes(node)
        if node.op_type == "Resize" and len(node.input) >= 2 and node.input[1] in init:
            mode = attr.get("coordinate_transformation_mode", b"half_pixel")
            if isinstance(mode, bytes):
                mode = mode.decode("ascii")
            if mode != "tf_crop_and_resize":
                result.append((index, 1, "Resize roi is ignored outside tf_crop_and_resize"))

        if node.op_type == "Pad" and len(node.input) >= 3 and node.input[2] in init:
            value = init[node.input[2]]
            if value.size == 1 and bool(np.all(value == 0)):
                result.append((index, 2, "Pad constant_value equals default zero"))

        if node.op_type == "Slice":
            if len(node.input) >= 5 and node.input[4] in init:
                steps = init[node.input[4]]
                if steps.size and bool(np.all(steps == 1)):
                    result.append((index, 4, "Slice steps equal default ones"))
            if len(node.input) >= 4 and node.input[3] in init and node.input[1] in init:
                axes = init[node.input[3]].reshape(-1)
                starts = init[node.input[1]].reshape(-1)
                if np.array_equal(axes, np.arange(len(starts), dtype=axes.dtype)):
                    result.append((index, 3, "Slice axes equal default leading axes"))

        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            if len(node.input) >= 3 and node.input[2] in init:
                zero = init[node.input[2]]
                if zero.dtype == np.uint8 and bool(np.all(zero == 0)):
                    result.append((index, 2, f"{node.op_type} uint8 zero_point equals default"))

        if node.op_type in {"Conv", "ConvTranspose"}:
            if len(node.input) >= 3 and node.input[2] in init:
                bias = init[node.input[2]]
                if bool(np.all(bias == 0)):
                    result.append((index, 2, f"{node.op_type} bias is exactly zero"))

        if node.op_type == "Gemm" and len(node.input) >= 3 and node.input[2] in init:
            bias = init[node.input[2]]
            if bool(np.all(bias == 0)):
                result.append((index, 2, "Gemm C is exactly zero"))
    return result


def structural(model: onnx.ModelProto) -> tuple[bool, str | None]:
    try:
        if not try_candidate._validate_ops_and_shapes(model):
            return False, "try_candidate structural validator rejected"
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def profile(model: onnx.ModelProto, task: int) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"optdefault_{task:03d}_", dir="/tmp") as work:
        return scoring.score_and_verify(
            copy.deepcopy(model), task, work, label="candidate", require_correct=True
        )


def main() -> None:
    costs = target_costs()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "authority": AUTHORITY.name,
        "authority_sha256": digest(AUTHORITY.read_bytes()),
        "scope": sorted(costs),
        "excluded": sorted(EXCLUDED),
        "rows": [],
        "winners": [],
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in sorted(costs):
            source = archive.read(f"task{task:03d}.onnx")
            original = onnx.load_model_from_string(source)
            plans = opportunities(original)
            task_row: dict[str, Any] = {
                "task": task,
                "authority_cost": costs[task],
                "opportunity_count": len(plans),
                "attempts": [],
            }
            for node_index, input_index, reason in plans:
                candidate, meta = rewrite_once(
                    original, node_index, input_index, reason
                )
                item: dict[str, Any] = {"meta": meta}
                ok, error = structural(candidate)
                item["structural_ok"] = ok
                item["structural_error"] = error
                if not ok:
                    task_row["attempts"].append(item)
                    continue
                try:
                    scored = profile(candidate, task)
                except Exception as exc:  # noqa: BLE001
                    scored = None
                    item["profile_error"] = f"{type(exc).__name__}: {exc}"
                item["profile"] = scored
                if scored is None or int(scored["cost"]) >= costs[task]:
                    task_row["attempts"].append(item)
                    continue

                gold_ok, mismatch = try_candidate._verify_gold(candidate, task)
                margin_ok, minimum = try_candidate._check_margin(candidate, task)
                item.update(
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
                    task_row["attempts"].append(item)
                    continue

                data = candidate.SerializeToString()
                candidate_cost = int(scored["cost"])
                path = OUT / (
                    f"task{task:03d}_cost{candidate_cost}_"
                    f"{node_index:03d}_{candidate.graph.node[node_index].op_type}.onnx"
                )
                path.write_bytes(data)
                winner = {
                    "task": task,
                    "authority_cost": costs[task],
                    "candidate_cost": candidate_cost,
                    "score_gain": math.log(costs[task] / candidate_cost),
                    "path": str(path.relative_to(ROOT)),
                    "sha256": digest(data),
                    **item,
                }
                task_row["attempts"].append(winner)
                report["winners"].append(winner)
            report["rows"].append(task_row)
            if plans:
                print(
                    json.dumps(
                        {
                            "task": task,
                            "cost": costs[task],
                            "plans": len(plans),
                            "wins": sum("path" in row for row in task_row["attempts"]),
                        }
                    ),
                    flush=True,
                )
    report["winner_count"] = len(report["winners"])
    report["total_gain"] = sum(row["score_gain"] for row in report["winners"])
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
