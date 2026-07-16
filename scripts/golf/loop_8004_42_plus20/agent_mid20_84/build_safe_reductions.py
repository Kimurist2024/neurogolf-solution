#!/usr/bin/env python3
"""Build and screen conservative exact mechanical reductions for the 20-task lane.

This script never promotes a candidate.  It operates on the exact authority
members and writes isolated models/evidence below this directory only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8005.17.zip"
COSTS_JSON = ROOT / "scripts/golf/loop_8004_42_plus20/current_costs_8004_50.json"
TARGETS = (374, 250, 62, 8, 275, 112, 168, 109, 160, 99, 279, 345, 245, 37, 297, 14, 92, 397, 394, 398)

# These tasks have documented private-zero/unsound authority history.  Exact
# behavioral preservation is not a proof of the hidden true rule, so this lane
# records opportunities but will never admit them.
PRIVATE_OR_UNSOUND = {112, 168}

sys.path.insert(0, str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
from screen_all import known_dual, static_audit  # noqa: E402
from harvest import actual_screen  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def init_key(item: onnx.TensorProto) -> tuple[Any, ...]:
    """Bit-stable key independent of the initializer name."""
    clone = copy.deepcopy(item)
    clone.name = ""
    return (int(item.data_type), tuple(item.dims), clone.SerializeToString())


def remove_now_unused_initializers(model: onnx.ModelProto) -> list[str]:
    live = {name for node in model.graph.node for name in node.input if name}
    live.update(value.name for value in model.graph.input)
    live.update(value.name for value in model.graph.output)
    removed = [item.name for item in model.graph.initializer if item.name not in live]
    if removed:
        keep = [item for item in model.graph.initializer if item.name not in set(removed)]
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep)
    return removed


def dedupe_initializers(model: onnx.ModelProto) -> list[dict[str, str]]:
    protected = {value.name for value in model.graph.input}
    protected.update(value.name for value in model.graph.output)
    first: dict[tuple[Any, ...], str] = {}
    replacements: dict[str, str] = {}
    for item in model.graph.initializer:
        if item.name in protected:
            continue
        key = init_key(item)
        if key in first:
            replacements[item.name] = first[key]
        else:
            first[key] = item.name
    if not replacements:
        return []
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]
    keep = [item for item in model.graph.initializer if item.name not in replacements]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return [{"removed": old, "reused": new} for old, new in sorted(replacements.items())]


def remove_zero_conv_biases(model: onnx.ModelProto) -> list[dict[str, str]]:
    initializers = {item.name: item for item in model.graph.initializer}
    actions: list[dict[str, str]] = []
    for node in model.graph.node:
        index = 8 if node.op_type == "QLinearConv" else (2 if node.op_type in {"Conv", "ConvTranspose"} else None)
        if index is None or len(node.input) <= index or not node.input[index]:
            continue
        name = node.input[index]
        item = initializers.get(name)
        if item is None:
            continue
        try:
            values = numpy_helper.to_array(item)
        except Exception:
            continue
        if values.size and np.all(values == 0):
            # Bias is the last optional input for all three operators.
            del node.input[index:]
            actions.append({"op": node.op_type, "output": node.output[0], "bias": name})
    remove_now_unused_initializers(model)
    return actions


def bypass_identity_nodes(model: onnx.ModelProto) -> list[dict[str, str]]:
    graph_outputs = {value.name for value in model.graph.output}
    actions: list[dict[str, str]] = []
    remove_ids: set[int] = set()
    # Repeatedly remove identities because a chain may become directly reusable.
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            if id(node) in remove_ids or node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
                continue
            source, target = node.input[0], node.output[0]
            if not source or not target or target in graph_outputs:
                continue
            for consumer in model.graph.node:
                if id(consumer) in remove_ids or consumer is node:
                    continue
                for index, name in enumerate(consumer.input):
                    if name == target:
                        consumer.input[index] = source
            remove_ids.add(id(node))
            actions.append({"output": target, "source": source})
            changed = True
    if remove_ids:
        keep = [node for node in model.graph.node if id(node) not in remove_ids]
        del model.graph.node[:]
        model.graph.node.extend(keep)
        remove_now_unused_initializers(model)
    return actions


def apply_kind(base: onnx.ModelProto, kind: str) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(base)
    detail: dict[str, Any] = {}
    if kind in {"unused", "combined"}:
        detail["unused_initializers"] = remove_now_unused_initializers(model)
    if kind in {"dedupe", "combined"}:
        detail["deduplicated_initializers"] = dedupe_initializers(model)
    if kind in {"zero_bias", "combined"}:
        detail["removed_zero_biases"] = remove_zero_conv_biases(model)
    if kind in {"identity", "combined"}:
        detail["bypassed_identities"] = bypass_identity_nodes(model)
    # Transformations can expose new unused constants.
    if kind == "combined":
        detail["final_unused_initializers"] = remove_now_unused_initializers(model)
    detail["action_count"] = sum(len(value) for key, value in detail.items() if isinstance(value, list))
    return model, detail


def main() -> int:
    out_dir = HERE / "reduced_candidates"
    out_dir.mkdir(exist_ok=True)
    costs = {int(k): int(v) for k, v in json.loads(COSTS_JSON.read_text())["costs"].items()}
    rows: list[dict[str, Any]] = []
    unique: set[tuple[int, str]] = set()
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TARGETS:
            base_data = archive.read(f"task{task:03d}.onnx")
            base = onnx.load_model_from_string(base_data)
            for kind in ("unused", "dedupe", "zero_bias", "identity", "combined"):
                model, actions = apply_kind(base, kind)
                if not actions["action_count"]:
                    continue
                data = model.SerializeToString()
                digest = sha256(data)
                if digest == sha256(base_data) or (task, digest) in unique:
                    continue
                unique.add((task, digest))
                path = out_dir / f"task{task:03d}_{kind}_{digest[:12]}.onnx"
                path.write_bytes(data)
                row: dict[str, Any] = {
                    "task": task,
                    "kind": kind,
                    "path": rel(path),
                    "sha256": digest,
                    "authority_cost": costs[task],
                    "actions": actions,
                    "policy_private_or_unsound": task in PRIVATE_OR_UNSOUND,
                }
                audit = static_audit(data, [rel(BASE_ZIP)], task)
                row["static_audit"] = audit
                if not audit["pass"]:
                    row.update(stage="static_reject", reasons=audit["reasons"])
                    rows.append(row)
                    continue
                actual = actual_screen(data, task)
                row["actual_cost"] = actual
                if actual is None or actual >= costs[task]:
                    row.update(stage="not_cheaper", reasons=["actual_cost_not_strictly_lower"])
                    rows.append(row)
                    continue
                row["gain"] = math.log(costs[task] / actual)
                if task in PRIVATE_OR_UNSOUND:
                    row.update(stage="policy_reject", reasons=["private_or_unsound_without_complete_true_rule_proof"])
                    rows.append(row)
                    continue
                dual = known_dual(task, data)
                row["known_dual"] = dual
                if any(
                    mode.get("wrong") or mode.get("errors") or mode.get("session_error") or not mode.get("right")
                    for mode in dual.values()
                ):
                    row.update(stage="known_reject", reasons=["known_dual_not_100_percent"])
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
            print("PRE_FRESH", row["task"], row["kind"], row["authority_cost"], row["actual_cost"], row["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
