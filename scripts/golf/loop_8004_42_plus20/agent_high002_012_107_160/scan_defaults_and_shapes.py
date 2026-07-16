#!/usr/bin/env python3
"""Audit exact default-attribute removals and truthful-shape rewrites."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (2, 12, 107)
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "high160_default_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high160_default_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def official(task: int, model: onnx.ModelProto, label: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"high160_{task:03d}_", dir="/tmp") as wd:
        result = scoring.score_and_verify(
            copy.deepcopy(model), task, wd, label=label, require_correct=False
        )
    if result is None:
        raise RuntimeError("score_and_verify returned None")
    return result


def exact_default_attribute_actions(model: onnx.ModelProto) -> list[tuple[int, str]]:
    actions: list[tuple[int, str]] = []
    for index, node in enumerate(model.graph.node):
        attrs = {attribute.name: attribute for attribute in node.attribute}
        if node.op_type == "Conv":
            # Kernel dimensions are exactly inferable from the weight initializer.
            if "kernel_shape" in attrs:
                actions.append((index, "kernel_shape"))
            if "group" in attrs and attrs["group"].i == 1:
                actions.append((index, "group"))
            if "auto_pad" in attrs and attrs["auto_pad"].s in (b"", b"NOTSET"):
                actions.append((index, "auto_pad"))
            if "strides" in attrs and all(value == 1 for value in attrs["strides"].ints):
                actions.append((index, "strides"))
            if "dilations" in attrs and all(value == 1 for value in attrs["dilations"].ints):
                actions.append((index, "dilations"))
            if "pads" in attrs and all(value == 0 for value in attrs["pads"].ints):
                actions.append((index, "pads"))
        if node.op_type == "Pad" and "mode" in attrs and attrs["mode"].s == b"constant":
            actions.append((index, "mode"))
        if node.op_type == "ReduceSum" and "keepdims" in attrs and attrs["keepdims"].i == 1:
            actions.append((index, "keepdims"))
        if node.op_type == "GatherND" and "batch_dims" in attrs and attrs["batch_dims"].i == 0:
            actions.append((index, "batch_dims"))
    return actions


def remove_attributes(model: onnx.ModelProto, actions: list[tuple[int, str]]) -> None:
    grouped: dict[int, set[str]] = {}
    for index, name in actions:
        grouped.setdefault(index, set()).add(name)
    for index, names in grouped.items():
        kept = [attribute for attribute in model.graph.node[index].attribute if attribute.name not in names]
        del model.graph.node[index].attribute[:]
        model.graph.node[index].attribute.extend(kept)


def set_shape(value_info: onnx.ValueInfoProto, shape: list[int]) -> None:
    tensor_type = value_info.type.tensor_type
    del tensor_type.shape.dim[:]
    for value in shape:
        tensor_type.shape.dim.add().dim_value = int(value)


def truthful_shape_variant(model: onnx.ModelProto, trace: dict[str, object]) -> tuple[onnx.ModelProto, list[str]]:
    candidate = copy.deepcopy(model)
    actual = trace.get("actual_shapes", {})
    changed = []
    for value_info in list(candidate.graph.value_info) + list(candidate.graph.output):
        shape = actual.get(value_info.name) if isinstance(actual, dict) else None
        if isinstance(shape, list) and all(isinstance(value, int) and value >= 0 for value in shape):
            old = [dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim]
            if old != shape:
                set_shape(value_info, shape)
                changed.append(value_info.name)
    return candidate, changed


def audit_variant(task: int, kind: str, model: onnx.ModelProto, base_cost: int) -> dict[str, object]:
    data = model.SerializeToString()
    row: dict[str, object] = {
        "task": task,
        "kind": kind,
        "sha256": digest(data),
        "serialized_bytes": len(data),
    }
    try:
        row["structural"] = SCAN.structural(copy.deepcopy(model))
    except Exception as exc:  # noqa: BLE001
        row["structural"] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        row["official_profile"] = official(task, model, f"high160_{task:03d}_{kind}")
    except Exception as exc:  # noqa: BLE001
        row["official_profile"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        row["runtime_shape_trace"] = AUDIT.direct_trace(task, data)
    except Exception as exc:  # noqa: BLE001
        row["runtime_shape_trace"] = {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}
    cost = row["official_profile"].get("cost")
    row["strict_lower"] = isinstance(cost, int) and cost < base_cost
    row["accepted"] = False
    row["reasons"] = []
    if not row["strict_lower"]:
        row["reasons"].append("not_strict_lower")
    if not row["structural"].get("pass", False):
        row["reasons"].append("structural_reject")
    if not row["runtime_shape_trace"].get("truthful", False):
        row["reasons"].append("runtime_shape_not_truthful")
    if not row["reasons"]:
        row["reasons"].append("known_dual_and_policy_gates_not_yet_run_fail_closed")
    return row


def main() -> int:
    if digest((ROOT / "submission.zip").read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    authority = json.loads((HERE / "authority_audit.json").read_text(encoding="utf-8"))
    rows = []
    for task in TASKS:
        base = onnx.load(HERE / f"current/task{task:03d}.onnx")
        base_cost = authority["tasks"][str(task)]["official_profile"]["cost"]
        actions = exact_default_attribute_actions(base)
        for index, action in enumerate(actions):
            candidate = copy.deepcopy(base)
            remove_attributes(candidate, [action])
            row = audit_variant(task, f"remove_default_attr_{action[0]}_{action[1]}", candidate, base_cost)
            row["attribute_actions"] = [list(action)]
            rows.append(row)
        if actions:
            candidate = copy.deepcopy(base)
            remove_attributes(candidate, actions)
            row = audit_variant(task, "remove_all_exact_default_attrs", candidate, base_cost)
            row["attribute_actions"] = [list(action) for action in actions]
            rows.append(row)
        trace = authority["tasks"][str(task)]["runtime_shape_trace"]
        candidate, changed = truthful_shape_variant(base, trace)
        if changed:
            row = audit_variant(task, "rewrite_value_info_to_runtime_shapes", candidate, base_cost)
            row["shape_names_changed"] = changed
            rows.append(row)
    report = {
        "authority_sha256": AUTHORITY_SHA256,
        "semantic_basis": {
            "default_attributes": "Only schema defaults or Conv kernel shape inferable from the immutable weight are removed.",
            "shape_rewrite": "Value-info declarations are replaced by witnessed runtime shapes; no computation is changed.",
        },
        "rows": rows,
        "strict_lower_truthful_survivors": [
            row for row in rows
            if row["strict_lower"] and row["structural"].get("pass")
            and row["runtime_shape_trace"].get("truthful")
        ],
    }
    (HERE / "defaults_and_shapes_scan.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(
            f"task{row['task']:03d} {row['kind']} cost={row['official_profile'].get('cost')} "
            f"struct={row['structural'].get('pass')} "
            f"truthful={row['runtime_shape_trace'].get('truthful')} lower={row['strict_lower']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
