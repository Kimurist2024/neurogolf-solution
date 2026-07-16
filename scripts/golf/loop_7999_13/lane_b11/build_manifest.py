#!/usr/bin/env python3
"""Assemble the reproducible B11 no-adoption manifest from audit evidence."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
TASKS = (264, 281, 300, 358, 376, 387, 392)


def load(name: str) -> dict[str, object]:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> None:
    inventory = load("baseline_inventory.json")
    shape_safety = load("baseline_shape_safety.json")
    history = load("history_scan.json")
    rejections = load("candidate_rejections.json")
    decisions = {
        264: "reject: exact member is shape-cloaked/default-ORT-invalid; sole cheaper local alternative fails both ORT modes",
        281: "no candidate: no local lower-bound improvement; exact member uses a 38-input giant Einsum and has runtime shape mismatches",
        300: "no candidate: the only sub-cost lower-bound record is the exact incumbent itself; it uses giant Einsum contractions and shape mismatches",
        358: "no candidate: no lower-bound local alternative; exact member uses a 44-input giant Einsum",
        376: "no adoption: clean cost-158 incumbent is at the 30-int32 Gather-index architecture floor; sole nominally cheaper variant fails full checker",
        387: "no candidate: no lower-bound local alternative; exact member has runtime shape-value mismatches",
        392: "reject: all five distinct lower-bound alternatives use prohibited lookup operators and fail the first known case in both ORT modes",
    }
    clean_baseline = {264: False, 281: False, 300: False, 358: False, 376: True, 387: False, 392: False}
    tasks: dict[str, object] = {}
    for task in TASKS:
        model = onnx.load(HERE / "baseline" / f"task{task:03d}.onnx")
        ops = Counter(node.op_type for node in model.graph.node)
        giant = [
            {"node_index": index, "inputs": len(node.input)}
            for index, node in enumerate(model.graph.node)
            if node.op_type == "Einsum" and len(node.input) >= 8
        ]
        shape_row = shape_safety[str(task)]
        trace = shape_row["runtime_shape_trace"]
        history_rows = history[str(task)]["models_with_declared_lower_bound_below_baseline"]
        alternatives = [row for row in history_rows if not row["is_exact_baseline"]]
        tasks[str(task)] = {
            "baseline": inventory["tasks"][str(task)],
            "baseline_op_histogram": dict(ops.most_common()),
            "baseline_giant_einsum_nodes": giant,
            "baseline_lookup_nodes": ops.get("TfIdfVectorizer", 0),
            "baseline_runtime_shape_mismatches": len(trace.get("declared_actual_mismatches", [])),
            "baseline_disable_all_one_case": shape_row["disable_all_one_case"],
            "baseline_default_one_case": shape_row["default_one_case"],
            "baseline_clean_under_campaign_gate": clean_baseline[task],
            "unique_local_models_screened": history[str(task)]["unique_local_models"],
            "strictly_lower_declared_bound_alternatives": len(alternatives),
            "decision": decisions[task],
        }
    manifest = {
        "campaign": "loop_7999_13_lane_b11",
        "baseline_score": 7999.13,
        "baseline_zip": inventory["baseline_zip"],
        "baseline_zip_sha256": inventory["baseline_zip_sha256"],
        "scope_tasks": list(TASKS),
        "policy": {
            "strict_improvement_only": True,
            "known_examples_all_required": True,
            "fresh_seed": 5000,
            "fresh_cases_per_ort_mode": 5000,
            "required_ort_modes": ["ORT_DISABLE_ALL", "default"],
            "errors_allowed": 0,
            "prohibited": [
                "giant Einsum",
                "lookup encoding",
                "metadata-only savings",
                "shape-value cloak",
                "undefined behavior",
                "nonstandard domain",
                "out-of-spec assumptions",
            ],
        },
        "history_screen": {
            "unique_models": sum(
                int(history[str(task)]["unique_local_models"]) for task in TASKS
            ),
            "source": "every local .onnx associated with the seven tasks, SHA-256 deduplicated",
        },
        "adopted_candidates": [],
        "aggregate_cost_delta": 0,
        "aggregate_score_delta": 0.0,
        "fresh5000_runs": [],
        "fresh5000_omission_reason": "No candidate survived the earlier structural/checker/known-case gates, so no model was eligible for adoption-scale fresh verification.",
        "tasks": tasks,
        "rejection_evidence": rejections,
        "evidence_files": [
            "baseline_inventory.json",
            "exact_graph_audit.json",
            "baseline_shape_safety.json",
            "history_scan.json",
            "candidate_rejections.json",
        ],
        "root_submission_mutations": [],
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"adopted": 0, "delta": 0, "screened": 300}, indent=2))


if __name__ == "__main__":
    main()
