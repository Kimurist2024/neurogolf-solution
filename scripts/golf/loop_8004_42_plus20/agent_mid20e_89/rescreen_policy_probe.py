#!/usr/bin/env python3
"""Re-open policy-only candidates and extract LB_PROBE_REQUIRED models.

Schema-invalid candidates remain hard rejects. Lookup/giant/private/nonfinite
lineage is treated as an LB-probe label rather than a permanent task ban, but
only after an actually strict-lower cost and complete known×4 evidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESCREEN = HERE / "rescreen.json"
ALLOWED_PROBE_REASONS = {
    "lookup", "giant_einsum", "private_zero_lineage", "nonfinite_initializer",
    "fresh_below_90", "fresh_below_95_or_runtime_error",
}

sys.path.insert(0, str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from screen_all import resolve_source  # noqa: E402
from harvest import actual_screen, known_score  # noqa: E402
from audit_candidates import runtime_shape_trace  # noqa: E402

SCAN_SPEC = importlib.util.spec_from_file_location("mid20e89_probe_quad", HERE / "scan_authority.py")
if SCAN_SPEC is None or SCAN_SPEC.loader is None:
    raise RuntimeError("cannot load known-quad scanner")
SCAN = importlib.util.module_from_spec(SCAN_SPEC)
SCAN_SPEC.loader.exec_module(SCAN)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve(row: dict) -> tuple[bytes, str]:
    for source in row["sources"]:
        data = resolve_source(source, int(row["task"]))
        if data is not None and sha256(data) == row["sha256"]:
            return data, source
    raise RuntimeError(f"cannot resolve task{row['task']:03d} {row['sha256']}")


def has_nonfinite(data: bytes) -> list[str]:
    model = onnx.load_model_from_string(data)
    names = []
    for item in model.graph.initializer:
        try:
            array = numpy_helper.to_array(item)
        except Exception:
            continue
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            names.append(item.name)
    return names


def main() -> int:
    audit_dir = HERE / "audit"
    audit_dir.mkdir(exist_ok=True)
    report = json.loads(RESCREEN.read_text())
    policy_rows = [row for row in report["rows"] if row["stage"] == "policy_reject"]
    reopened = []
    probes = []
    for row in policy_rows:
        reasons = set(row.get("reasons", []))
        item = {
            "task": row["task"],
            "sha256": row["sha256"],
            "sources": row["sources"],
            "policy_reasons": sorted(reasons),
            "static_floor": row.get("static_floor"),
            "authority_cost": row["current_actual_cost"],
            "strict_detail": row.get("strict_detail"),
        }
        if not reasons or not reasons.issubset(ALLOWED_PROBE_REASONS):
            item.update(decision="HARD_REJECT_SCHEMA_OR_NONPROBE_POLICY")
            reopened.append(item)
            continue
        if row.get("static_floor") is None or int(row["static_floor"]) >= int(row["current_actual_cost"]):
            item.update(decision="REJECT_STATIC_FLOOR_NOT_STRICTLY_LOWER")
            reopened.append(item)
            continue

        data, source = resolve(row)
        item["resolved_source"] = source
        item["nonfinite_initializers"] = has_nonfinite(data)
        actual = actual_screen(data, int(row["task"]))
        item["actual_cost"] = actual
        # Run the full diagnostics even if actual cost loses, because shape
        # cloaks can make the static floor look artificially attractive.
        official = known_score(data, int(row["task"]), True, f"mid20e89_policy_{row['sha256'][:8]}")
        item["official_like_score"] = official
        item["known_quad"] = SCAN.SCANNER.known_dual(int(row["task"]), data)
        try:
            item["runtime_shape_trace"] = runtime_shape_trace(
                int(row["task"]), onnx.load_model_from_string(data)
            )
        except Exception as exc:  # noqa: BLE001
            item["runtime_shape_trace_error"] = f"{type(exc).__name__}: {exc}"

        known_ok = all(
            not mode.get("wrong")
            and not mode.get("errors")
            and not mode.get("session_error")
            and bool(mode.get("right"))
            for mode in item["known_quad"].values()
        )
        trace = item.get("runtime_shape_trace") or {}
        truthful = not trace.get("declared_actual_mismatches") and not item.get("runtime_shape_trace_error")
        strict_lower = actual is not None and int(actual) < int(row["current_actual_cost"])
        official_lower = (
            isinstance(official, dict)
            and official.get("correct")
            and int(official["cost"]) < int(row["current_actual_cost"])
        )
        if strict_lower and official_lower and known_ok and truthful:
            item.update(
                decision="LB_PROBE_REQUIRED",
                gain=math.log(int(row["current_actual_cost"]) / int(official["cost"])),
            )
            probes.append(item)
        else:
            failures = []
            if not strict_lower:
                failures.append("actual_cost_not_strictly_lower")
            if not official_lower:
                failures.append("official_cost_or_correctness")
            if not known_ok:
                failures.append("known_quad")
            if not truthful:
                failures.append("shape_cloak")
            item.update(decision="REJECT_AFTER_POLICY_REOPEN", failures=failures)
        reopened.append(item)

    counts = Counter(item["decision"] for item in reopened)
    evidence = {
        "baseline_zip": report["baseline_zip"],
        "baseline_zip_sha256": report["baseline_zip_sha256"],
        "policy_reject_total": len(policy_rows),
        "allowed_probe_reasons": sorted(ALLOWED_PROBE_REASONS),
        "decision_counts": dict(counts),
        "rows": reopened,
    }
    (audit_dir / "policy_reopen.json").write_text(json.dumps(evidence, indent=2) + "\n")
    manifest = {
        "status": "LB_PROBE_REQUIRED_CANDIDATES" if probes else "NO_LB_PROBE_REQUIRED_CANDIDATE",
        "baseline_zip": report["baseline_zip"],
        "baseline_zip_sha256": report["baseline_zip_sha256"],
        "policy_reject_total": len(policy_rows),
        "policy_reason_reopened": sum(1 for item in reopened if set(item["policy_reasons"]).issubset(ALLOWED_PROBE_REASONS)),
        "hard_schema_or_nonprobe_reject": counts["HARD_REJECT_SCHEMA_OR_NONPROBE_POLICY"],
        "static_floor_not_lower": counts["REJECT_STATIC_FLOOR_NOT_STRICTLY_LOWER"],
        "fully_profiled_after_reopen": counts["REJECT_AFTER_POLICY_REOPEN"] + counts["LB_PROBE_REQUIRED"],
        "count": len(probes),
        "candidates": probes,
    }
    (HERE / "probe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "policy_reject_total", "policy_reason_reopened", "hard_schema_or_nonprobe_reject", "static_floor_not_lower", "fully_profiled_after_reopen", "count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
