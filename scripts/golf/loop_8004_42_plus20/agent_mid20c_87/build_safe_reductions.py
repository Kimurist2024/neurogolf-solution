#!/usr/bin/env python3
"""Build conservative exact/no-op reductions from the exact authority members."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8005.17.zip"
TARGETS = (51, 64, 29, 178, 123, 91, 124, 148, 199, 341, 357, 137, 355, 169, 316, 212, 301, 174, 153, 325)
PRIVATE_OR_UNSOUND = {178, 169, 174, 325}

# Reuse the already audited initializer/bias/Identity primitives.
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20_84/build_safe_reductions.py"
SPEC = importlib.util.spec_from_file_location("mid20c87_safe_primitives", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load safe reduction primitives")
SAFE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAFE)

sys.path.insert(0, str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
from screen_all import known_dual, static_audit  # noqa: E402
from harvest import actual_screen, known_score  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def tensor_shape(value: onnx.ValueInfoProto | None) -> tuple[int, ...] | None:
    if value is None or not value.type.HasField("tensor_type"):
        return None
    dims = value.type.tensor_type.shape.dim
    if any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in dims):
        return None
    return tuple(int(dim.dim_value) for dim in dims)


def bypass_noops(model: onnx.ModelProto) -> list[dict[str, Any]]:
    """Remove nodes whose output is exactly their data input."""
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    values = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    graph_outputs = {value.name for value in model.graph.output}
    remove_ids: set[int] = set()
    actions: list[dict[str, Any]] = []

    for node in model.graph.node:
        if len(node.output) != 1 or not node.output[0] or node.output[0] in graph_outputs:
            continue
        source: str | None = None
        reason: str | None = None
        if node.op_type == "Concat" and len(node.input) == 1:
            source, reason = node.input[0], "single_input_concat"
        elif node.op_type in {"Add", "Mul"} and len(node.input) == 2:
            neutral = 0 if node.op_type == "Add" else 1
            for index in (0, 1):
                array = initializers.get(node.input[index])
                if array is not None and array.size and np.all(array == neutral):
                    source, reason = node.input[1 - index], f"{node.op_type.lower()}_neutral"
                    break
        elif node.op_type in {"Sub", "Div"} and len(node.input) == 2:
            neutral = 0 if node.op_type == "Sub" else 1
            array = initializers.get(node.input[1])
            if array is not None and array.size and np.all(array == neutral):
                source, reason = node.input[0], f"{node.op_type.lower()}_right_neutral"
        elif node.op_type == "Transpose" and len(node.input) == 1:
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            perm = attrs.get("perm")
            if perm is not None and list(perm) == list(range(len(perm))):
                source, reason = node.input[0], "identity_transpose"
        elif node.op_type == "Cast" and len(node.input) == 1:
            before, after = values.get(node.input[0]), values.get(node.output[0])
            if before is not None and after is not None and before.type.tensor_type.elem_type == after.type.tensor_type.elem_type:
                source, reason = node.input[0], "same_dtype_cast"
        elif node.op_type == "Reshape" and len(node.input) >= 1:
            if tensor_shape(values.get(node.input[0])) == tensor_shape(values.get(node.output[0])):
                source, reason = node.input[0], "same_shape_reshape"
        elif node.op_type == "Pad" and len(node.input) >= 2:
            pads = initializers.get(node.input[1])
            if pads is not None and np.all(pads == 0):
                source, reason = node.input[0], "zero_pad"
        if source is None or not source:
            continue
        target = node.output[0]
        for consumer in model.graph.node:
            if consumer is node:
                continue
            for index, name in enumerate(consumer.input):
                if name == target:
                    consumer.input[index] = source
        remove_ids.add(id(node))
        actions.append({"op": node.op_type, "output": target, "source": source, "reason": reason})

    if remove_ids:
        keep = [node for node in model.graph.node if id(node) not in remove_ids]
        del model.graph.node[:]
        model.graph.node.extend(keep)
        SAFE.remove_now_unused_initializers(model)
    return actions


def transform(base: onnx.ModelProto, kind: str) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(base)
    detail: dict[str, Any] = {}
    if kind in {"unused", "combined"}:
        detail["unused_initializers"] = SAFE.remove_now_unused_initializers(model)
    if kind in {"dedupe", "combined"}:
        detail["deduplicated_initializers"] = SAFE.dedupe_initializers(model)
    if kind in {"zero_bias", "combined"}:
        detail["removed_zero_biases"] = SAFE.remove_zero_conv_biases(model)
    if kind in {"identity", "combined"}:
        detail["bypassed_identities"] = SAFE.bypass_identity_nodes(model)
    if kind in {"noops", "combined"}:
        detail["bypassed_noops"] = bypass_noops(model)
    if kind == "combined":
        detail["final_unused_initializers"] = SAFE.remove_now_unused_initializers(model)
    detail["action_count"] = sum(len(value) for value in detail.values() if isinstance(value, list))
    return model, detail


def main() -> int:
    out_dir = HERE / "reduced_candidates"
    out_dir.mkdir(exist_ok=True)
    authority = json.loads((HERE / "authority_costs.json").read_text())["costs"]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TARGETS:
            base_data = archive.read(f"task{task:03d}.onnx")
            base = onnx.load_model_from_string(base_data)
            for kind in ("unused", "dedupe", "zero_bias", "identity", "noops", "combined"):
                model, actions = transform(base, kind)
                if not actions["action_count"]:
                    continue
                data = model.SerializeToString()
                digest = sha256(data)
                if digest == sha256(base_data) or (task, digest) in seen:
                    continue
                seen.add((task, digest))
                path = out_dir / f"task{task:03d}_{kind}_{digest[:12]}.onnx"
                path.write_bytes(data)
                row: dict[str, Any] = {
                    "task": task,
                    "kind": kind,
                    "path": rel(path),
                    "sha256": digest,
                    "authority_cost": int(authority[str(task)]),
                    "actions": actions,
                }
                audit = static_audit(data, [rel(BASE_ZIP)], task)
                row["static_audit"] = audit
                if task in PRIVATE_OR_UNSOUND:
                    row.update(stage="policy_reject", reasons=["private_or_unsound_without_complete_true_rule_proof"])
                elif not audit["pass"]:
                    row.update(stage="static_reject", reasons=audit["reasons"])
                else:
                    actual = actual_screen(data, task)
                    row["actual_screen_cost"] = actual
                    if actual is None or actual >= int(authority[str(task)]):
                        row.update(stage="actual_reject", reasons=["actual_cost_not_strictly_lower"])
                    else:
                        profile = known_score(data, task, True, f"mid20c87_mech_{task}_{digest[:8]}")
                        row["official_like_score"] = profile
                        if not profile or not profile.get("correct") or int(profile["cost"]) >= int(authority[str(task)]):
                            row.update(stage="known_reject", reasons=["known_not_complete_or_not_cheaper"])
                        else:
                            row["gain"] = math.log(int(authority[str(task)]) / int(profile["cost"]))
                            row["known_dual"] = known_dual(task, data)
                            if any(
                                mode.get("wrong") or mode.get("errors") or mode.get("session_error") or not mode.get("right")
                                for mode in row["known_dual"].values()
                            ):
                                row.update(stage="known_dual_reject", reasons=["known_dual_not_100_percent"])
                            else:
                                row.update(stage="pre_fresh", reasons=[])
                rows.append(row)
    report = {
        "baseline": rel(BASE_ZIP),
        "baseline_sha256": sha256(BASE_ZIP.read_bytes()),
        "targets": list(TARGETS),
        "candidate_count": len(rows),
        "stage_counts": dict(Counter(row["stage"] for row in rows)),
        "rows": rows,
    }
    (HERE / "mechanical_reductions.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "stage_counts": report["stage_counts"]}, indent=2))
    for row in rows:
        if row["stage"] == "pre_fresh":
            print("PRE_FRESH", row["task"], row["path"], row["official_like_score"]["cost"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
