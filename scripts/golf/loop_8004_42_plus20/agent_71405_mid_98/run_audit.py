#!/usr/bin/env python3
"""Deep audit of the selected others/71405 mid-cost candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
PRIVATE_HIGH_RISK = {96, 138, 157, 209}
FILES = (
    (89, "others/71405/task089_improved(1).onnx"),
    (96, "others/71405/task096_improved(1).onnx"),
    (107, "others/71405/task107_improved (1).onnx"),
    (117, "others/71405/task117_best605_rechecked.onnx"),
    (125, "others/71405/task125_improved(1).onnx"),
    (125, "others/71405/task125_improved_v2.onnx"),
    (138, "others/71405/task138_improved.onnx"),
    (156, "others/71405/task156_improved.onnx"),
    (157, "others/71405/task157_improved(1).onnx"),
    (165, "others/71405/task165_further_improved.onnx"),
    (209, "others/71405/task209_further_improved.onnx"),
)

sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
from harvest import known_score  # noqa: E402

HELPER_PATH = ROOT / "scripts/golf/loop_8004_42_plus20/agent_expand20i_94/screen_incremental.py"
SPEC = importlib.util.spec_from_file_location("lane98_helpers", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import lane94 audit helpers")
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profile_twice(data: bytes, task: int, kind: str) -> dict:
    runs = [
        known_score(data, task, False, f"lane98_{kind}_{task}_{run}")
        for run in (1, 2)
    ]
    return {
        "runs": runs,
        "identical": len({json.dumps(item, sort_keys=True) for item in runs}) == 1,
        "profile": runs[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    (HERE / "audit").mkdir(parents=True, exist_ok=True)
    (HERE / "candidates").mkdir(exist_ok=True)
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("submission.zip does not match the approved 8006.61 authority")
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority = {task: archive.read(f"task{task:03d}.onnx") for task, _ in FILES}

    authority_path = HERE / "audit/authority_profiles_2x.json"
    if authority_path.exists():
        cached_authority = json.loads(authority_path.read_text())
        if cached_authority.get("authority_zip_sha256") != AUTHORITY_SHA256:
            raise RuntimeError("cached authority profile hash mismatch")
        authority_profiles = cached_authority["tasks"]
    else:
        authority_profiles = {}
        for task in sorted(authority):
            authority_profiles[str(task)] = {
                "sha256": digest(authority[task]),
                **profile_twice(authority[task], task, "authority"),
            }
        authority_path.write_text(json.dumps({
            "authority_zip_sha256": AUTHORITY_SHA256,
            "tasks": authority_profiles,
        }, indent=2) + "\n")

    rows = []
    qualified = []
    for task, source in FILES:
        data = (ROOT / source).read_bytes()
        sha = digest(data)
        candidate_profile = profile_twice(data, task, "candidate")
        auth = authority_profiles[str(task)]["profile"]
        cand = candidate_profile["profile"]
        row = {
            "task": task,
            "source": source,
            "sha256": sha,
            "bytes": len(data),
            "authority_sha256": authority_profiles[str(task)]["sha256"],
            "authority_profile": auth,
            "candidate_profile_2x": candidate_profile,
            "candidate_profile": cand,
            "strict_lower": bool(cand and auth and int(cand["cost"]) < int(auth["cost"])),
            "gain": math.log(int(auth["cost"]) / int(cand["cost"])) if cand and auth and int(cand["cost"]) < int(auth["cost"]) else 0.0,
            "structural_audit": HELPERS.structural_audit(data),
            "private_high_risk": task in PRIVATE_HIGH_RISK,
            "exact_sha_lb_history": "NO_EXACT_SHA_LB_RECORD_FOUND",
        }
        if not candidate_profile["identical"] or not cand:
            row["classification"] = "REJECT_OFFICIAL_PROFILE_UNSTABLE_OR_FAILED"
            rows.append(row)
            continue
        if not row["strict_lower"]:
            row["classification"] = "REJECT_NOT_STRICTLY_LOWER"
            rows.append(row)
            continue
        if not row["structural_audit"]["pass"]:
            row["classification"] = "REJECT_STRUCTURE_SCHEMA_UB"
            rows.append(row)
            continue
        row["known_four"] = HELPERS.known_four(task, data)
        if not cand.get("correct") or not HELPERS.known_complete(row["known_four"]):
            runtime_failure = any(
                item.get("session_error") or item.get("errors", 0)
                for item in row["known_four"].values()
            )
            row["classification"] = "REJECT_RUNTIME_CONFIG" if runtime_failure else "REJECT_KNOWN"
            rows.append(row)
            continue
        try:
            row["runtime_shape_trace"] = HELPERS.runtime_shape_trace(task, onnx.load_model_from_string(data))
        except Exception as exc:  # noqa: BLE001
            row["runtime_shape_trace_error"] = f"{type(exc).__name__}: {exc}"
            row["classification"] = "REJECT_RUNTIME_SHAPE_TRACE"
            rows.append(row)
            continue
        if row["runtime_shape_trace"]["declared_actual_mismatches"]:
            row["classification"] = "REJECT_SHAPE_CLOAK"
            rows.append(row)
            continue
        row["classification"] = "FRESH_PENDING"
        row["data"] = data
        qualified.append(row)
        rows.append(row)

    checkpoint_rows = []
    for row in rows:
        item = dict(row)
        item.pop("data", None)
        checkpoint_rows.append(item)
    (HERE / "audit/pre_fresh_screen.json").write_text(json.dumps({
        "authority_zip_sha256": AUTHORITY_SHA256,
        "classification_counts": dict(Counter(row["classification"] for row in rows)),
        "qualified_tasks": sorted({int(row["task"]) for row in qualified}),
        "rows": checkpoint_rows,
    }, indent=2) + "\n")
    if args.screen_only:
        print(json.dumps({
            "stage": "PRE_FRESH_COMPLETE",
            "classification_counts": dict(Counter(row["classification"] for row in rows)),
            "qualified_tasks": sorted({int(row["task"]) for row in qualified}),
        }, indent=2))
        return 0

    fresh_reports = {}
    for task in sorted({int(row["task"]) for row in qualified}):
        task_rows = [row for row in qualified if int(row["task"]) == task]
        report = HELPERS.fresh_two_seed(task, task_rows)
        fresh_reports[str(task)] = report
        for row in task_rows:
            config_rates = []
            seed_rows = []
            for run in report["runs"]:
                stats = run["candidates"][row["sha256"]]
                rates = {name: item["right"] / 500 for name, item in stats.items()}
                config_rates.extend(rates.values())
                seed_rows.append({"seed": run["seed"], "rates": rates})
            row["fresh_two_seed"] = {
                "count_per_seed": 500,
                "seeds": seed_rows,
                "minimum_config_rate": min(config_rates),
                "maximum_config_rate": max(config_rates),
            }
            row["classification"] = "LB_PROBE_REQUIRED"
            isolated = HERE / "candidates" / f"task{task:03d}_{row['sha256'][:12]}_cost{row['candidate_profile']['cost']}.onnx"
            isolated.write_bytes(row["data"])
            row["isolated_path"] = str(isolated.relative_to(ROOT))
            row.pop("data", None)
    (HERE / "audit/fresh_two_seed.json").write_text(json.dumps(fresh_reports, indent=2) + "\n")

    probe_rows = []
    for row in rows:
        if row["classification"] != "LB_PROBE_REQUIRED":
            continue
        priority = "LOW" if row["private_high_risk"] or row["fresh_two_seed"]["minimum_config_rate"] < 0.90 else "NORMAL"
        probe_rows.append({
            "task": row["task"],
            "sha256": row["sha256"],
            "path": row["isolated_path"],
            "source": row["source"],
            "authority_cost": int(row["authority_profile"]["cost"]),
            "candidate_cost": int(row["candidate_profile"]["cost"]),
            "gain": row["gain"],
            "fresh_two_seed": row["fresh_two_seed"],
            "private_high_risk": row["private_high_risk"],
            "probe_priority": priority,
            "exact_sha_lb_history": row["exact_sha_lb_history"],
            "decision": "LB_PROBE_REQUIRED_NOT_FIXED",
        })
    probe_rows.sort(key=lambda item: (item["probe_priority"] == "LOW", -item["gain"], item["task"], item["sha256"]))
    screen = {
        "authority_zip_sha256": AUTHORITY_SHA256,
        "file_count": len(FILES),
        "task_count": len(set(task for task, _ in FILES)),
        "classification_counts": dict(Counter(row["classification"] for row in rows)),
        "rows": rows,
    }
    (HERE / "audit/full_screen.json").write_text(json.dumps(screen, indent=2) + "\n")
    (HERE / "audit/exact_sha_lb_history.json").write_text(json.dumps({
        "rule": "exact SHA match only; task-level private history is a risk flag, not an exact verdict",
        "search_surfaces": ["docs/golf", "scripts JSON/Markdown", "campaign MEMORY.md"],
        "exact_matches": [],
        "rows": [{"task": row["task"], "sha256": row["sha256"], "classification": row["exact_sha_lb_history"]} for row in rows],
    }, indent=2) + "\n")
    (HERE / "probe_manifest.json").write_text(json.dumps({
        "status": "LB_PROBE_REQUIRED_CANDIDATES" if probe_rows else "NO_LB_PROBE_REQUIRED_CANDIDATE",
        "authority_zip_sha256": AUTHORITY_SHA256,
        "count": len(probe_rows),
        "candidates": probe_rows,
    }, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(json.dumps({
        "status": "NO_FIXED_WINNER",
        "authority_zip_sha256": AUTHORITY_SHA256,
        "count": 0,
        "candidates": [],
        "reason": "No candidate has exact LB-white evidence; private-risk tasks are explicitly forbidden from fixed adoption.",
    }, indent=2) + "\n")
    print(json.dumps({
        "classification_counts": screen["classification_counts"],
        "probe_count": len(probe_rows),
        "probe_tasks": dict(Counter(item["task"] for item in probe_rows)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
