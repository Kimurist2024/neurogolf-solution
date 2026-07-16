#!/usr/bin/env python3
"""Strict structural, cost, known-all, and truthful-shape audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
COSTS = {319: 1003, 367: 2179}

sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
from harvest import known_score  # noqa: E402

helper_path = ROOT / "scripts/golf/loop_8004_42_plus20/agent_expand20i_94/screen_incremental.py"
spec = importlib.util.spec_from_file_location("lane107_audit_helpers", helper_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load audit helpers")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    manifest = json.loads((HERE / "audit/build_manifest.json").read_text())
    rows = []
    for source in manifest["candidates"]:
        task = int(source["task"])
        path = ROOT / source["path"]
        data = path.read_bytes()
        if digest(data) != source["sha256"]:
            raise RuntimeError(f"candidate hash drift: {path}")
        model = onnx.load_model_from_string(data)
        names = [item.name for item in model.graph.initializer]
        row = dict(source)
        row.update({
            "authority_cost": COSTS[task],
            "contains_task319_fixed_correction": task == 319 and any(
                "corr_" in name or "corr" in name for name in names
            ),
            "structural_audit": audit.structural_audit(data),
        })
        try:
            row["official_profile"] = known_score(
                data, task, False, f"sound319367_{task}_{source['label']}_{source['sha256'][:8]}"
            )
        except Exception as exc:  # noqa: BLE001
            row["official_profile_error"] = f"{type(exc).__name__}: {exc}"
            row["official_profile"] = None
        try:
            row["known_four"] = audit.known_four(task, data)
        except Exception as exc:  # noqa: BLE001
            row["known_four_error"] = f"{type(exc).__name__}: {exc}"
            row["known_four"] = {}
        try:
            trace = audit.runtime_shape_trace(task, model)
            row["runtime_shape_trace"] = trace
            row["truthful_shape"] = not trace["declared_actual_mismatches"]
        except Exception as exc:  # noqa: BLE001
            row["runtime_shape_trace_error"] = f"{type(exc).__name__}: {exc}"
            row["truthful_shape"] = False

        profile = row["official_profile"]
        row["candidate_cost"] = int(profile["cost"]) if profile else None
        row["strictly_lower"] = bool(profile and int(profile["cost"]) < COSTS[task])
        row["known_complete"] = audit.known_complete(row["known_four"])
        if not row["structural_audit"]["pass"]:
            row["decision"] = "REJECT_STRUCTURE_SCHEMA_UB"
        elif not profile or not profile.get("correct") or not row["known_complete"]:
            row["decision"] = "REJECT_KNOWN_OR_RUNTIME"
        elif not row["truthful_shape"]:
            row["decision"] = "REJECT_SHAPE_CLOAK"
        elif not row["strictly_lower"]:
            row["decision"] = "REJECT_NOT_STRICTLY_LOWER"
        elif row["contains_task319_fixed_correction"]:
            row["decision"] = "REJECT_FIXED_CORRECTION_PROVENANCE"
        else:
            row["gain"] = math.log(COSTS[task] / row["candidate_cost"])
            row["decision"] = "FRESH_PENDING"
        rows.append(row)
        print(
            f"AUDIT task{task:03d} {source['label']} cost={row['candidate_cost']} "
            f"known={row['known_complete']} truthful={row['truthful_shape']} "
            f"decision={row['decision']}",
            flush=True,
        )

    report = {
        "authority_costs": COSTS,
        "candidate_count": len(rows),
        "fresh_pending_count": sum(row["decision"] == "FRESH_PENDING" for row in rows),
        "rows": rows,
    }
    (HERE / "audit/candidate_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "candidate_count": report["candidate_count"],
        "fresh_pending_count": report["fresh_pending_count"],
        "decisions": {decision: sum(row["decision"] == decision for row in rows)
                      for decision in sorted({row["decision"] for row in rows})},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
