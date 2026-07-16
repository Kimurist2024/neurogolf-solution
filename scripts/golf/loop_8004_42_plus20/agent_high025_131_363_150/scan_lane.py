#!/usr/bin/env python3
"""Fail-closed inventory and exact mechanical scan for tasks 025/131/363."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (25, 131, 363)
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXACT = load_module(
    "high150_exact",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8008_exact_white102/scan_exact.py",
)
SCAN = load_module(
    "high150_scan",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high150_audit",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def action_count(detail: dict[str, object]) -> int:
    return sum(len(value) for value in detail.values() if isinstance(value, list))


def transform(base: onnx.ModelProto, kind: str) -> tuple[onnx.ModelProto, dict[str, object]]:
    model = copy.deepcopy(base)
    detail: dict[str, object] = {}
    if kind in ("cleanup", "combined", "normalized_combined"):
        detail["dead_nodes"] = EXACT.remove_dead_nodes(model)
        detail["unused_initializers"] = EXACT.remove_unused_initializers(model)
        detail["dead_value_info"] = EXACT.remove_dead_value_info(model)
    if kind in ("dedupe", "combined", "normalized_combined"):
        detail["deduplicated_initializers"] = EXACT.dedupe_initializers(model)
    if kind in ("optional", "combined", "normalized_combined"):
        detail["removed_optional_outputs"] = EXACT.remove_optional_outputs(model)
    if kind in ("noops", "combined", "normalized_combined"):
        detail["bypassed_noops"] = EXACT.bypass_noops(model)
    if kind in ("cse", "combined", "normalized_combined"):
        detail["common_subexpressions"] = EXACT.common_subexpressions(model)
    if kind in ("fold", "combined", "normalized_combined"):
        detail["constant_folds"] = EXACT.constant_fold(model)
    if kind in ("combined", "normalized_combined"):
        detail["second_noops"] = EXACT.bypass_noops(model)
        detail["second_cse"] = EXACT.common_subexpressions(model)
        detail["final_dead_nodes"] = EXACT.remove_dead_nodes(model)
        detail["final_unused_initializers"] = EXACT.remove_unused_initializers(model)
    if kind in ("normalize", "normalized_combined"):
        detail["cleared_value_info"] = EXACT.clear_value_info(model)
    return model, detail


def safe_trace(task: int, data: bytes) -> dict[str, object]:
    try:
        return AUDIT.direct_trace(task, data)
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    if digest((ROOT / "submission.zip").read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority changed")
    out: dict[str, object] = {
        "authority": "submission.zip",
        "authority_sha256": AUTHORITY_SHA256,
        "tasks": {},
        "strict_lower": [],
    }
    kinds = (
        "cleanup", "dedupe", "noops", "cse", "optional", "fold",
        "combined", "normalize", "normalized_combined",
    )
    for task in TASKS:
        path = HERE / f"baseline/task{task:03d}.onnx"
        data = path.read_bytes()
        base = onnx.load_model_from_string(data)
        base_profile = SCAN.official_cost(data, f"high150_task{task:03d}_base")
        task_row: dict[str, object] = {
            "sha256": digest(data),
            "serialized_bytes": len(data),
            "official_profile": base_profile,
            "structural": SCAN.structural(copy.deepcopy(base)),
            "runtime_shape_trace": safe_trace(task, data),
            "graph_inventory": SCAN.graph_inventory(copy.deepcopy(base)),
            "exact_rows": [],
        }
        seen = {digest(data)}
        for kind in kinds:
            candidate, actions = transform(base, kind)
            if action_count(actions) == 0:
                continue
            candidate_data = candidate.SerializeToString()
            sha = digest(candidate_data)
            if sha in seen:
                continue
            seen.add(sha)
            static = SCAN.structural(copy.deepcopy(candidate))
            row: dict[str, object] = {
                "kind": kind,
                "sha256": sha,
                "actions": actions,
                "structural": static,
                "strict_lower": False,
            }
            if static.get("pass"):
                profile = SCAN.official_cost(candidate_data, f"high150_task{task:03d}_{kind}")
                row["official_profile"] = profile
                row["strict_lower"] = 0 <= profile["cost"] < base_profile["cost"]
                row["runtime_shape_trace"] = safe_trace(task, candidate_data)
                if row["strict_lower"]:
                    candidate_path = HERE / f"candidates/task{task:03d}_{kind}_{sha[:12]}.onnx"
                    candidate_path.write_bytes(candidate_data)
                    row["path"] = str(candidate_path.relative_to(ROOT))
                    out["strict_lower"].append({"task": task, **row})
            task_row["exact_rows"].append(row)
        out["tasks"][str(task)] = task_row
        print(
            f"task{task:03d} cost={base_profile['cost']} "
            f"structural={task_row['structural'].get('pass')} "
            f"truthful={task_row['runtime_shape_trace'].get('truthful')} "
            f"exact_variants={len(task_row['exact_rows'])}",
            flush=True,
        )
    (HERE / "audit/inventory_exact.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"strict_lower={len(out['strict_lower'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
