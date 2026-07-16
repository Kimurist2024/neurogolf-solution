#!/usr/bin/env python3
"""Current-only exact mechanical scan for task125/165/268."""

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
TASKS = (125, 165, 268)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXACT = load_module(
    "high135_exact_transforms",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8008_exact_white102/scan_exact.py",
)
SCAN = load_module(
    "high135_cost_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high135_runtime_tools",
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


def safe_declared_cost(data: bytes, label: str) -> dict[str, object]:
    try:
        return SCAN.official_cost(data, label)
    except Exception as exc:  # noqa: BLE001
        return {"memory": -1, "params": -1, "cost": -1, "error": f"{type(exc).__name__}: {exc}"}


def safe_trace(task: int, data: bytes) -> dict[str, object]:
    try:
        return AUDIT.direct_trace(task, data)
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    candidates = HERE / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    audit = json.loads((HERE / "authority_audit.json").read_text(encoding="utf-8"))
    kinds = (
        "cleanup", "dedupe", "noops", "cse", "optional", "fold",
        "combined", "normalize", "normalized_combined",
    )
    report: dict[str, object] = {"tasks": {}, "preliminary_lower": []}
    for task in TASKS:
        base_data = (HERE / f"current/task{task:03d}.onnx").read_bytes()
        base = onnx.load_model_from_string(base_data)
        base_cost = audit["tasks"][str(task)]["official_profile"]
        seen = {digest(base_data)}
        rows = []
        for kind in kinds:
            model, detail = transform(base, kind)
            if action_count(detail) == 0:
                continue
            data = model.SerializeToString()
            sha = digest(data)
            if sha in seen:
                continue
            seen.add(sha)
            static = SCAN.structural(copy.deepcopy(model))
            row: dict[str, object] = {
                "task": task, "kind": kind, "sha256": sha,
                "actions": detail, "structural": static,
                "status": "STRUCTURAL_REJECT",
            }
            if static.get("pass"):
                profile = safe_declared_cost(data, f"high135_task{task:03d}_{kind}")
                row["declared_shape_cost_profile"] = profile
                row["preliminary_lower"] = profile["cost"] >= 0 and profile["cost"] < base_cost["cost"]
                row["runtime_shape_trace"] = safe_trace(task, data)
                row["status"] = "PRELIMINARY_LOWER_NEEDS_OFFICIAL_AUDIT" if row["preliminary_lower"] else "NOT_LOWER"
                if row["preliminary_lower"]:
                    path = candidates / f"task{task:03d}_{kind}_{sha[:12]}.onnx"
                    path.write_bytes(data)
                    row["path"] = str(path.relative_to(ROOT))
                    report["preliminary_lower"].append(row)
            rows.append(row)
            print(
                f"task{task:03d} {kind} actions={action_count(detail)} "
                f"declared_cost={row.get('declared_shape_cost_profile', {}).get('cost')} "
                f"lower={row.get('preliminary_lower', False)} "
                f"truthful={row.get('runtime_shape_trace', {}).get('truthful')}",
                flush=True,
            )
        report["tasks"][str(task)] = {
            "baseline_sha256": digest(base_data),
            "baseline_official_profile": base_cost,
            "rows": rows,
        }
    (HERE / "exact_scan.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"preliminary_lower={len(report['preliminary_lower'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
