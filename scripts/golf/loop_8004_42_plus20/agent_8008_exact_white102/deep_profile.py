#!/usr/bin/env python3
"""Directly profile every structurally admitted exact transform.

`actual_screen` is intentionally only a fast prefilter.  This script records the
official competition-profile result even when that prefilter returned None, so a
potential exact win cannot be lost to a local screening false negative.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
from harvest import known_score  # noqa: E402

helper_path = ROOT / "scripts/golf/loop_8004_42_plus20/agent_expand20i_94/screen_incremental.py"
spec = importlib.util.spec_from_file_location("lane102_deep_helpers", helper_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load audit helpers")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main() -> int:
    scan_path = HERE / "audit/mechanical_scan.json"
    scan = json.loads(scan_path.read_text())
    promising_rejected = {(182, "noops"), (191, "noops"), (209, "noops")}
    selected = [
        row for row in scan["rows"]
        if row["actions"]["semantic_action_count"] > 0
        and (
            row["structural_audit"]["pass"]
            or (row["task"], row["kind"]) in promising_rejected
        )
    ]
    rows = []
    for row in selected:
        data = (ROOT / row["path"]).read_bytes()
        result = {
            "task": row["task"],
            "kind": row["kind"],
            "path": row["path"],
            "sha256": row["sha256"],
            "authority_cost": row["authority_cost"],
            "actual_screen_cost": row.get("actual_screen_cost"),
            "structural_pass": row["structural_audit"]["pass"],
            "structural_hard_failures": row["structural_audit"]["hard_failures"],
        }
        try:
            result["official_profile"] = known_score(
                data,
                row["task"],
                False,
                f"lane102_direct_{row['task']}_{row['kind']}_{row['sha256'][:8]}",
            )
        except Exception as exc:  # noqa: BLE001
            result["official_profile_error"] = f"{type(exc).__name__}: {exc}"
        try:
            trace = audit.runtime_shape_trace(row["task"], onnx.load_model_from_string(data))
            result["shape_trace"] = {
                "declared_actual_mismatch_count": len(trace["declared_actual_mismatches"]),
                "declared_actual_mismatches": trace["declared_actual_mismatches"],
            }
        except Exception as exc:  # noqa: BLE001
            result["shape_trace_error"] = f"{type(exc).__name__}: {exc}"
        profile = result.get("official_profile")
        result["admitted"] = bool(
            profile
            and profile.get("correct")
            and int(profile["cost"]) < row["authority_cost"]
            and result.get("shape_trace", {}).get("declared_actual_mismatch_count") == 0
        )
        rows.append(result)
        print(
            f"PROFILE task{row['task']:03d} {row['kind']} "
            f"profile={profile} admitted={result['admitted']}",
            flush=True,
        )

    report = {
        "source": "audit/mechanical_scan.json",
        "selected_count": len(rows),
        "admitted_count": sum(row["admitted"] for row in rows),
        "rows": rows,
    }
    (HERE / "audit/deep_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("selected_count", "admitted_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
