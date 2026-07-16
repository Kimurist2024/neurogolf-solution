#!/usr/bin/env python3
"""Reopen policy-only SHAs and build the LB_PROBE_REQUIRED manifest."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ALLOWED = {"lookup", "giant_einsum", "private_zero_lineage", "nonfinite_initializer", "fresh_below_90", "fresh_below_95_or_runtime_error"}

sys.path.insert(0, str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from screen_all import resolve_source  # noqa: E402
from harvest import actual_screen, known_score, known_worker, run_bounded, screen_worker  # noqa: E402
from audit_candidates import runtime_shape_trace  # noqa: E402

SPEC = importlib.util.spec_from_file_location("expand20h92_known4", HERE / "scan_authority.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load scan wrapper")
SCAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCAN)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve(row):
    for source in row["sources"]:
        data = resolve_source(source, int(row["task"]))
        if data is not None and sha256(data) == row["sha256"]:
            return data, source
    raise RuntimeError(f"unresolved task{row['task']:03d} {row['sha256']}")


def known4_worker(job):
    digest, task, data = job
    try:
        return {"sha256": digest, "task": task, "known_four": SCAN.known_four(task, data)}
    except BaseException as exc:  # noqa: BLE001
        return {"sha256": digest, "task": task, "error": f"{type(exc).__name__}: {exc}"}


def known_complete(result):
    return result and all(
        mode.get("right")
        and not mode.get("wrong")
        and not mode.get("errors")
        and not mode.get("session_error")
        for mode in result.values()
    )


def main() -> int:
    audit_dir = HERE / "audit"
    audit_dir.mkdir(exist_ok=True)
    report = json.loads((HERE / "rescreen.json").read_text())
    rows = report["rows"]
    policy = [row for row in rows if row["stage"] == "policy_reject"]
    audit_rows = []
    jobs = []
    data_by_sha = {}
    row_by_sha = {}
    for row in policy:
        reasons = set(row.get("reasons", []))
        item = {
            "task": row["task"], "sha256": row["sha256"], "sources": row["sources"],
            "policy_reasons": sorted(reasons), "static_floor": row.get("static_floor"),
            "authority_cost": row["current_actual_cost"], "strict_detail": row.get("strict_detail"),
        }
        audit_rows.append(item)
        if not reasons or not reasons.issubset(ALLOWED):
            item["decision"] = "HARD_REJECT_SCHEMA_UB_OR_STRICT_INFERENCE"
            continue
        if row.get("static_floor") is None or int(row["static_floor"]) >= int(row["current_actual_cost"]):
            item["decision"] = "REJECT_STATIC_FLOOR_NOT_LOWER"
            continue
        data, source = resolve(row)
        item["resolved_source"] = source
        data_by_sha[row["sha256"]] = data
        row_by_sha[row["sha256"]] = (row, item)
        jobs.append((row["sha256"], int(row["task"]), data))

    actual_results = run_bounded(jobs, screen_worker, max_workers=4, timeout=30.0, label="PROBE_ACTUAL")
    lower_jobs = []
    for result in actual_results:
        digest = result.get("sha256")
        if digest not in row_by_sha:
            continue
        row, item = row_by_sha[digest]
        item["actual_screen_result"] = result
        cost = result.get("cost")
        if cost is None or int(cost) >= int(row["current_actual_cost"]):
            item["decision"] = "REJECT_ACTUAL_NOT_LOWER_OR_RUNTIME"
            continue
        item["actual_screen_cost"] = int(cost)
        lower_jobs.append((digest, int(row["task"]), data_by_sha[digest]))

    official_results = run_bounded(lower_jobs, known_worker, max_workers=4, timeout=60.0, label="PROBE_OFFICIAL")
    known4_results = run_bounded(lower_jobs, known4_worker, max_workers=4, timeout=60.0, label="PROBE_KNOWN4")
    official_by_sha = {item.get("sha256"): item for item in official_results}
    known4_by_sha = {item.get("sha256"): item for item in known4_results}
    probes = []
    for digest, task, data in lower_jobs:
        row, item = row_by_sha[digest]
        official = official_by_sha.get(digest, {})
        quad_result = known4_by_sha.get(digest, {})
        profile = official.get("result")
        quad = quad_result.get("known_four")
        item["official_result"] = official
        item["known_four_result"] = quad_result
        try:
            trace = runtime_shape_trace(task, onnx.load_model_from_string(data))
            item["runtime_shape_trace"] = trace
        except Exception as exc:  # noqa: BLE001
            trace = None
            item["runtime_shape_trace_error"] = f"{type(exc).__name__}: {exc}"
        official_lower = bool(profile and profile.get("correct") and int(profile["cost"]) < int(row["current_actual_cost"]))
        truthful = bool(trace is not None and not trace["declared_actual_mismatches"])
        if official_lower and known_complete(quad) and truthful:
            candidate = {
                "task": task, "sha256": digest, "path": item["resolved_source"],
                "authority_cost": row["current_actual_cost"], "candidate_cost": int(profile["cost"]),
                "gain": math.log(int(row["current_actual_cost"]) / int(profile["cost"])),
                "probe_reasons": item["policy_reasons"], "known_four": quad,
                "runtime_shape_trace": trace, "source": "policy_reopen",
            }
            item["decision"] = "LB_PROBE_REQUIRED"
            probes.append(candidate)
        else:
            failures = []
            if not official_lower: failures.append("official_not_correct_and_lower")
            if not known_complete(quad): failures.append("known_four")
            if not truthful: failures.append("shape_truth")
            item.update(decision="REJECT_AFTER_FULL_REOPEN", failures=failures)

    # Clean structural candidates with complete known evidence are preserved as
    # probes whenever fresh is imperfect; sampled generator accuracy is not a
    # proof of LB whiteness.
    for row in rows:
        if row["stage"] not in {"fresh500_pass", "fresh500_reject"}:
            continue
        fresh = row.get("fresh_dual") or {}
        if not known_complete(row.get("known_dual")):
            continue
        candidate = {
            "task": row["task"], "sha256": row["sha256"], "path": row["isolated_candidate"],
            "authority_cost": row["current_actual_cost"], "candidate_cost": row["actual_cost"],
            "gain": row["gain"], "probe_reasons": ["fresh_not_100_percent"],
            "fresh500": fresh, "known_four": row["known_dual"],
            "runtime_shape_trace": row["runtime_shape_trace"], "source": "clean_known_complete",
        }
        probes.append(candidate)

    # SHA-deduplicate and keep every distinct net; ordering is deterministic.
    probes = sorted({item["sha256"]: item for item in probes}.values(), key=lambda x: (x["task"], x["candidate_cost"], x["sha256"]))
    decisions = Counter(item["decision"] for item in audit_rows)
    (audit_dir / "policy_reopen.json").write_text(json.dumps({"count": len(audit_rows), "decision_counts": dict(decisions), "rows": audit_rows}, indent=2) + "\n")
    manifest = {
        "status": "LB_PROBE_REQUIRED_CANDIDATES" if probes else "NO_LB_PROBE_REQUIRED_CANDIDATE",
        "baseline_zip": report["baseline_zip"], "baseline_zip_sha256": report["baseline_zip_sha256"],
        "policy_total": len(policy), "policy_reason_reopened": sum(1 for item in audit_rows if set(item["policy_reasons"]) and set(item["policy_reasons"]).issubset(ALLOWED)),
        "policy_static_lower_jobs": len(jobs), "policy_actual_lower_jobs": len(lower_jobs),
        "policy_decision_counts": dict(decisions), "count": len(probes), "candidates": probes,
    }
    (HERE / "probe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "policy_total": manifest["policy_total"], "policy_reason_reopened": manifest["policy_reason_reopened"], "policy_static_lower_jobs": len(jobs), "policy_actual_lower_jobs": len(lower_jobs), "probe_count": len(probes), "probe_tasks": Counter(item["task"] for item in probes)}, default=dict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
