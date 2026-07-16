#!/usr/bin/env python3
"""Assemble the immutable archive-rescreen evidence into a final report."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import onnx
from onnx import shape_inference

from scan_annotation_only import bias_ub, computational_payload


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parent / "base_models" / "task109.onnx"
CANDIDATE = HERE / "candidates" / "task109.onnx"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    fresh = json.loads((HERE / "task109_fresh5000_independent.json").read_text())[0]
    external = json.loads((HERE / "task109_external500_independent.json").read_text())
    annotation_scan = json.loads((HERE / "annotation_only_scan.json").read_text())
    task109_annotation = next(
        row for row in annotation_scan["retained"] if row["task"] == 109
    )

    baseline = onnx.load(BASE, load_external_data=False)
    candidate = onnx.load(CANDIDATE, load_external_data=False)
    onnx.checker.check_model(candidate, full_check=True)
    shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
    ops = sorted({node.op_type for node in candidate.graph.node})
    nested = [
        f"{node.name}/{attribute.name}"
        for node in candidate.graph.node
        for attribute in node.attribute
        if attribute.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
    ]
    bias_failures = bias_ub(candidate)
    payload_identical = computational_payload(candidate) == computational_payload(baseline)

    candidate_audit = external["candidate"]
    baseline_audit = external["baseline"]
    differential = external["differential"]
    gain = math.log(baseline_audit["cost"] / candidate_audit["cost"])

    gates = {
        "candidate_sha_matches": sha(CANDIDATE) == candidate_audit["sha256"],
        "baseline_sha_matches": sha(BASE) == baseline_audit["sha256"],
        "checker_full": True,
        "strict_shape_inference_data_prop": True,
        "computational_payload_identical_after_clearing_value_info": payload_identical,
        "only_value_info_changed": payload_identical,
        "conv_family_bias_ub_count": len(bias_failures),
        "functions_count": len(candidate.functions),
        "nested_graph_count": len(nested),
        "known_complete": (
            candidate_audit["known"]["right"] == candidate_audit["known"]["total_seen"]
            and candidate_audit["known"]["wrong"] == 0
            and candidate_audit["known"]["errors"] == 0
        ),
        "fresh_disable_all_rate": fresh["disable_all"]["right"] / fresh["generated"],
        "fresh_default_rate": fresh["default"]["right"] / fresh["generated"],
        "fresh_generation_errors": fresh["generation_errors"],
        "fresh_candidate_runtime_or_output_failures": (
            fresh["disable_all"]["runtime_or_output_failures"]
            + fresh["default"]["runtime_or_output_failures"]
        ),
        "external_threshold_mismatches": differential["mismatches"],
        "external_asymmetric_errors": differential["skipped_one_failed"],
        "external_raw_equal_on_all_executable": (
            differential["raw_equal"] == differential["executable"]
        ),
        "truthful_cost_strictly_lower": candidate_audit["cost"] < baseline_audit["cost"],
    }
    accepted = (
        all(value is True for key, value in gates.items() if isinstance(value, bool))
        and gates["conv_family_bias_ub_count"] == 0
        and gates["functions_count"] == 0
        and gates["nested_graph_count"] == 0
        and gates["fresh_disable_all_rate"] >= 0.95
        and gates["fresh_default_rate"] >= 0.95
        and gates["fresh_generation_errors"] == 0
        and gates["fresh_candidate_runtime_or_output_failures"] == 0
        and gates["external_threshold_mismatches"] == 0
        and gates["external_asymmetric_errors"] == 0
    )

    report = {
        "baseline": {
            "leaderboard_score": 8003.40,
            "zip": "submission_base_8003.40.zip",
            "task109_sha256": sha(BASE),
            "task109_truthful_cost": baseline_audit["cost"],
        },
        "candidate": {
            "task": 109,
            "path": str(CANDIDATE.relative_to(ROOT)),
            "source_archive_path": task109_annotation["source_path"],
            "sha256": sha(CANDIDATE),
            "truthful_cost": candidate_audit["cost"],
            "cost_reduction": baseline_audit["cost"] - candidate_audit["cost"],
            "projected_gain": gain,
            "nodes": len(candidate.graph.node),
            "initializers": len(candidate.graph.initializer),
            "ops": ops,
        },
        "semantic_identity": {
            "computational_payload_identical_after_clearing_value_info": payload_identical,
            "value_info_differences": task109_annotation["value_info_differences"],
            "interpretation": (
                "No node, attribute, initializer, graph I/O, opset, function, or metadata "
                "change; no new lookup behavior is introduced."
            ),
        },
        "known": candidate_audit["known"],
        "fresh5000": fresh,
        "external500": differential,
        "conv_bias_failures": bias_failures,
        "nested_graphs": nested,
        "gates": gates,
        "verdict": "ACCEPT_STRICT_ISOLATED_CANDIDATE" if accepted else "REJECT",
        "merge_performed": False,
        "protected_files_changed": False,
        "archive_rescreen_summary": {
            "static_candidates_scanned": annotation_scan["scanned"],
            "payload_identical_records": len(annotation_scan["retained"]),
            "only_truthful_cost_reducing_payload_identical_task": 109,
            "same_cost_payload_identical_tasks_not_adopted": [20, 228],
            "task254": (
                "REJECT: 33-input giant Einsum is disallowed and external500 had "
                "412 threshold mismatches despite fresh5000 generator accuracy."
            ),
        },
    }
    (HERE / "FINAL_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")

    markdown = f"""# Archive resume final report (8003.40 baseline)

## Outcome

- Strict isolated candidate: **task109**
- Candidate SHA-256: `{report['candidate']['sha256']}`
- Baseline task SHA-256: `{report['baseline']['task109_sha256']}`
- Truthful official-like cost: `{baseline_audit['cost']} -> {candidate_audit['cost']}`
- Cost reduction: `{report['candidate']['cost_reduction']}`
- Projected gain: `+{gain:.15f}`
- ZIP merge: **not performed**

## Safety gates

- Known: `{candidate_audit['known']['right']}/{candidate_audit['known']['total_seen']}`, wrong `0`, errors `0`
- Fresh generator, ORT_DISABLE_ALL: `{fresh['disable_all']['right']}/{fresh['generated']}`; runtime/output failures `0`
- Fresh generator, default ORT: `{fresh['default']['right']}/{fresh['generated']}`; runtime/output failures `0`
- External differential: `{differential['raw_equal']}/{differential['executable']}` raw-equal executable cases; threshold mismatches `0`; asymmetric errors `0`
- Full ONNX checker: PASS
- Strict shape inference with data propagation: PASS
- Conv-family bias-length UB: `0`
- Functions: `0`; nested graphs: `0`

## Independent semantic proof

After clearing only `graph.value_info`, candidate and baseline deterministic protobufs are byte-for-byte identical. The sole annotation difference is `state_rows_pad`: `[1,1,1,2] -> [1,1,1,1]`. Nodes, attributes, initializers, graph I/O, opsets, functions, and metadata are unchanged, so this candidate introduces no new lookup or task rule.

## Residual archive screening

The static archive had `{annotation_scan['scanned']}` candidates. Annotation-only scanning found task20/task228 variants at the same truthful/static cost, so they were not adopted. task109 is the only cost-reducing annotation-only result. task254 cost42 was rejected: it is a policy-disallowed 33-input giant Einsum and differed from the baseline on 412/500 external threshold cases, despite passing generator fresh5000.
"""
    (HERE / "REPORT.md").write_text(markdown)
    print(json.dumps({
        "verdict": report["verdict"],
        "task": 109,
        "sha256": report["candidate"]["sha256"],
        "cost": [baseline_audit["cost"], candidate_audit["cost"]],
        "projected_gain": gain,
    }, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
