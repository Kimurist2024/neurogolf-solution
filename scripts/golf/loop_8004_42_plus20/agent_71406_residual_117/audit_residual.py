#!/usr/bin/env python3
"""Fail-closed audit of the residual strict-lower 71406/71409 candidates."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
INVENTORY = HERE / "inventory_screen.json"
EXPECTED_TASK = 382
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_module(
    "residual117_runtime_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)
CONV = load_module("residual117_conv_checker", ROOT / "scripts/golf/check_conv_bias.py")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def model_shape(model: onnx.ModelProto) -> dict[str, list[int | None]]:
    return {
        value.name: [
            int(dim.dim_value) if dim.HasField("dim_value") else None
            for dim in value.type.tensor_type.shape.dim
        ]
        for value in list(model.graph.input) + list(model.graph.output)
    }


def strict_checks(data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    result: dict[str, Any] = {
        "full_checker": False,
        "strict_shape_inference": False,
        "strict_shape_inference_data_prop": False,
        "declared_io_shapes": model_shape(model),
        "conv_family_bias_ub": [],
        "conv_family_bias_ub_count": 0,
    }
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        result["full_checker"] = True
    except Exception as exc:  # noqa: BLE001
        result["full_checker_error"] = f"{type(exc).__name__}: {exc}"
    try:
        onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True)
        result["strict_shape_inference"] = True
    except Exception as exc:  # noqa: BLE001
        result["strict_shape_inference_error"] = f"{type(exc).__name__}: {exc}"
    try:
        onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        result["strict_shape_inference_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        result["strict_shape_inference_data_prop_error"] = f"{type(exc).__name__}: {exc}"
    findings = CONV.check_model(copy.deepcopy(model))
    result["conv_family_bias_ub"] = [list(item) for item in findings]
    result["conv_family_bias_ub_count"] = len(findings)
    result["pass"] = bool(
        result["full_checker"]
        and result["strict_shape_inference"]
        and result["strict_shape_inference_data_prop"]
        and not findings
    )
    return result


def main() -> int:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority hash changed")
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    residual = inventory["strict_lower"]
    if len(residual) != 1 or residual[0]["task"] != EXPECTED_TASK:
        raise RuntimeError(f"unexpected residual set: {[row['task'] for row in residual]}")

    scan_row = residual[0]
    candidate_path = ROOT / scan_row["sources"][0]
    candidate = candidate_path.read_bytes()
    if digest(candidate) != scan_row["sha256"]:
        raise RuntimeError("candidate changed after inventory")
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority = archive.read(f"task{EXPECTED_TASK:03d}.onnx")

    static = strict_checks(candidate)
    runtime_shape = AUDIT.direct_trace(EXPECTED_TASK, candidate)
    known: dict[str, Any] = {}
    for disable, threads, label in CONFIGS:
        known[label] = AUDIT.known_config(EXPECTED_TASK, authority, candidate, disable, threads)

    all_known_candidate = all(
        row.get("candidate_right") == row.get("total")
        and row.get("runtime_errors", {}).get("candidate") == 0
        and row.get("nonfinite_values", {}).get("candidate") == 0
        for row in known.values()
    )
    all_known_authority = all(
        row.get("baseline_right") == row.get("total")
        and row.get("runtime_errors", {}).get("baseline") == 0
        and row.get("nonfinite_values", {}).get("baseline") == 0
        for row in known.values()
    )
    raw_equivalent = all(row.get("raw_equal") == row.get("total") for row in known.values())
    threshold_equivalent = all(row.get("threshold_equal") == row.get("total") for row in known.values())

    reasons: list[str] = []
    if not static["pass"]:
        reasons.append("static_checker_or_ub_gate_failed")
    if not runtime_shape.get("truthful", False):
        reasons.append("runtime_shapes_not_truthful")
    if not all_known_candidate:
        reasons.append("known_corpus_candidate_failed")
    if not all_known_authority:
        reasons.append("immutable_authority_known_failed")
    if not raw_equivalent:
        reasons.append("not_raw_equivalent_to_immutable_authority")
    if not threshold_equivalent:
        reasons.append("not_threshold_equivalent_to_immutable_authority")

    exact_current_output_proof = bool(
        scan_row["same_canonical_graph_as_authority"]
        or (raw_equivalent and runtime_shape.get("truthful", False))
    )
    true_rule_proof = False
    if not exact_current_output_proof:
        reasons.append("no_exact_current_output_proof")
    if not true_rule_proof:
        reasons.append("no_true_rule_proof")

    # Fresh testing is deliberately gated behind truthful runtime shapes and an
    # exact/true-rule proof. Sampling cannot rescue a shape-cloaked model.
    fresh = {
        "status": "not_run_fail_closed_before_fresh",
        "reason": "runtime shape truth and exact/true-rule admission gates did not pass",
    }
    accepted = not reasons
    verdict = "ACCEPT_EXACT" if accepted else "REJECT_NO_SAFE_EXACT_CANDIDATE"
    report = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "inventory": str(INVENTORY.relative_to(ROOT)),
        "candidate": {
            "task": EXPECTED_TASK,
            "path": str(candidate_path.relative_to(ROOT)),
            "sha256": digest(candidate),
            "authority_member_sha256": digest(authority),
            "candidate_profile": scan_row["candidate_profile"],
            "authority_profile": scan_row["authority_profile"],
            "cost_reduction": scan_row["cost_reduction"],
            "same_sha_as_authority": scan_row["same_sha_as_authority"],
            "same_canonical_graph_as_authority": scan_row["same_canonical_graph_as_authority"],
            "private_zero_history": scan_row["private_zero_history"],
            "static": static,
            "runtime_shape_trace": runtime_shape,
            "known_four_configs": known,
            "known_summary": {
                "candidate_perfect": all_known_candidate,
                "authority_perfect": all_known_authority,
                "raw_equivalent": raw_equivalent,
                "threshold_equivalent": threshold_equivalent,
            },
            "proof": {
                "exact_current_output": exact_current_output_proof,
                "true_rule": true_rule_proof,
                "note": "No reproducible transformation proof links this byte-distinct graph to the immutable member.",
            },
            "fresh": fresh,
            "history": {
                "same_named_older_model": "others/2/1203/task382_further_improved.onnx",
                "same_named_older_sha256": "3dd420479c491e37816dfa8e1c6629fb3e8552cf21d320ff14d1d9e9cf7ab24d",
                "same_named_older_cost": 817,
                "same_named_older_outcome": "historical complete-known scan rejected task382 family",
                "prior_cost814_sha256": "ac0d47cfa37effc8453f77a8498a0a8516ef63ab610ed6e4fb49afef485dee29",
                "prior_cost814_outcome": "fresh3000 passed but rejected by current truthful-shape policy: declared [1,10,1,1], runtime [1,10,30,30]",
            },
            "reasons": reasons,
            "accepted": accepted,
            "disposition": "winner" if accepted else "rejected_shape_cloak_not_quarantine",
        },
        "winners": [EXPECTED_TASK] if accepted else [],
        "summary": {
            "strict_lower_candidates": 1,
            "winner_count": int(accepted),
            "quarantine_count": 0,
            "rejected_count": int(not accepted),
            "verdict": verdict,
        },
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "authority": report["authority"],
        "authority_sha256": AUTHORITY_SHA256,
        "source_root": "others/71406",
        "inventory": report["inventory"],
        "audit": str((HERE / "audit.json").relative_to(ROOT)),
        "standalone_onnx_files": inventory["discovery"]["standalone_onnx_files"],
        "unique_task_sha_pairs": inventory["discovery"]["unique_task_sha_pairs"],
        "status_counts": inventory["status_counts"],
        "winners": [],
        "winner_count": 0,
        "promotion_performed": False,
        "verdict": verdict,
    }
    if accepted:
        manifest["winners"].append(report["candidate"])
        manifest["winner_count"] = 1
    (HERE / "winner_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
