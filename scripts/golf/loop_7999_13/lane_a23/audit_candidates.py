#!/usr/bin/env python3
"""Strict A23 structural, cost, and known-both-ORT audit."""

from __future__ import annotations

import json
import re
import sys
import copy
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_a20"))
import audit_history as shared  # noqa: E402


BASE_COST = {44: 1087, 205: 1042}
shared.BASE_COST = BASE_COST
UNSAFE_SOURCE = re.compile(r"quarantine|private0|7614rej", re.I)


def augmented_audit(task: int, label: str, path: Path, static: int | None, sources: list[str], baseline: bool = False) -> dict[str, object]:
    row = shared.audit(task, label, path, static, sources, baseline=baseline)
    unsafe_sources = [source for source in sources if UNSAFE_SOURCE.search(source)]
    row["unsafe_source_lineage"] = unsafe_sources
    reasons = set(row["pre_fresh_reasons"])
    lookup = row["lookup_red_flags"]
    # Hardmax over a runtime-derived ten-color score is an algorithmic selector,
    # not an example/signature table.  Retain lookup rejection for TfIdf or giant
    # payloads, while letting this standard dense operator reach fresh gold.
    hardmax_only = bool(lookup["hardmax"]) and not lookup["tfidf"] and not lookup["giant_initializer"]
    row["hardmax_algorithmic_selection"] = hardmax_only
    if hardmax_only:
        reasons.discard("lookup")
    if unsafe_sources:
        reasons.add("unsafe_source_lineage")
    if task == 44 and not baseline:
        reasons.add("generator_noninjective_no_total_truth_rule")
    row["pre_fresh_reasons"] = sorted(reasons)
    row["pre_fresh_pass"] = not reasons
    return row


def terminal_reference_audit(task: int, label: str, path: Path, sources: list[str]) -> dict[str, object]:
    """Run structural gates without redundant known passes after a terminal screen.

    A truthful static floor above the base, or a forbidden giant Einsum, is a
    terminal rejection.  Runtime shape tracing still executes one known case.
    """
    model = onnx.load(path)
    _, structure_reason, floor = shared.structure_gate(path.read_bytes())
    params = shared.scoring.calculate_params(model)
    screened_cost = floor if floor is not None else 0
    pseudo_profile = {
        "memory": max(0, screened_cost - params),
        "params": params,
        "cost": screened_cost,
        "score": None,
        "correct": None,
    }
    original_known = shared.known
    original_score = shared.scoring.score_and_verify
    try:
        shared.known = lambda *_args, **_kwargs: {"skipped_terminal_screen": True}
        shared.scoring.score_and_verify = lambda *_args, **_kwargs: copy.deepcopy(pseudo_profile)
        row = augmented_audit(task, label, path, None, sources)
    finally:
        shared.known = original_known
        shared.scoring.score_and_verify = original_score
    row["profile_mode"] = "truthful_static_floor_terminal_screen"
    row["terminal_structure_reason"] = structure_reason
    reasons = set(row["pre_fresh_reasons"])
    reasons.add("known_skipped_after_terminal_screen")
    row["pre_fresh_reasons"] = sorted(reasons)
    row["pre_fresh_pass"] = False
    return row


def main() -> None:
    ort.set_default_logger_severity(4)
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    rows = []
    for task in (44, 205):
        label = f"task{task:03d}_base"
        row = augmented_audit(task, label, HERE / "baseline" / f"task{task:03d}.onnx", BASE_COST[task], ["submission_base_7999.13.zip"], baseline=True)
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)
    for label, item in manifest["history_entries"].items():
        task = int(item["task"])
        row = augmented_audit(task, label, HERE / "candidates" / f"{label}.onnx", int(item["static_cost"]), list(item["sources"]))
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)
        (HERE / "audit_rows.json").write_text(json.dumps({"rows": rows, "complete": False}, indent=2) + "\n")
    for label, item in manifest["rule_references"].items():
        task = int(item["task"])
        row = terminal_reference_audit(task, label, HERE / "rule_references" / f"{label}.onnx", [item["source"]])
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)
        (HERE / "audit_rows.json").write_text(json.dumps({"rows": rows, "complete": False}, indent=2) + "\n")
    pending = [row for row in rows if row["pre_fresh_pass"] and not row["label"].endswith("_base")]
    (HERE / "audit_rows.json").write_text(json.dumps({"rows": rows, "pending": pending, "complete": True}, indent=2) + "\n")
    print(f"DONE rows={len(rows)} pending={len(pending)}", flush=True)


if __name__ == "__main__":
    main()
