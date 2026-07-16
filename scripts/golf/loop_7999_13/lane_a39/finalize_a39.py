#!/usr/bin/env python3
"""Finalize the A39 dead-node / dead-output audit without mutating authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
LANE = Path(__file__).resolve().parent
AUTHORITY = ROOT / "submission_base_8000.46.zip"
EXPECTED_SHA256 = "74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534"


def load(name: str):
    return json.loads((LANE / name).read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


authority_sha = sha256(AUTHORITY)
assert authority_sha == EXPECTED_SHA256, (authority_sha, EXPECTED_SHA256)

scan = load("scan_build_manifest.json")
named = load("named_dead_prune_probes.json")
omissions = load("multi_output_omission_probes.json")
anatomy = load("split_replacement_anatomy.json")

assert scan["candidate_count"] == 0
assert scan["errors"] == []
assert named["accepted"] == []

finding_tasks = scan["tasks_with_dead_nodes_or_multi_outputs"]
dead_node_tasks = sorted(
    item["task"] for item in finding_tasks if item.get("dead_nodes")
)
assert dead_node_tasks == [39, 89, 122, 183], dead_node_tasks

named_summary = {
    "task039": {
        "node": "Equal keep_bg_equal",
        "exclusion": "CenterCropPad lineage",
        "pruned_cost": 41,
        "known_dual": "264/264 runtime errors on both ORT modes",
        "reason": "dead node is an allocator barrier; Slice buffer-reuse mismatch after pruning",
    },
    "task089": {
        "node": "ReduceMax keep_red_big",
        "exclusion": "CenterCropPad lineage",
        "pruned_cost": None,
        "known_dual": "default ORT session creation fails; historical disabled-mode probe errors on all known cases",
        "reason": "dead node is an allocator/shape barrier; not an executable candidate",
    },
    "task122": {
        "node": "GreaterOrEqual d_keep",
        "exclusion": "CenterCropPad lineage",
        "pruned_cost": 102,
        "known_dual": "266/266 disabled-mode errors; 266/266 default-mode errors after pruning",
        "reason": "dead node is an allocator barrier; shape/buffer reuse breaks after pruning",
    },
    "task183": {
        "node": "Min hold_u8",
        "exclusion": "lookup lineage (ScatterElements)",
        "pruned_cost": 91,
        "known_dual": "265/265 runtime errors on both ORT modes",
        "reason": "dead node is an allocator barrier; Resize buffer-reuse mismatch after pruning",
    },
}

multi_output_summary = {
    "task019": {
        "op": "Split",
        "unused_output": "ca0",
        "empty_output_probe": "checker/strict pass, ORT process exits by SIGSEGV (-11)",
        "decision": "reject as undefined/unsafe runtime behavior",
    },
    "task124": {
        "op": "Split",
        "unused_output": "r3",
        "empty_output_probe": "checker/strict pass, ORT process exits by SIGSEGV (-11)",
        "decision": "reject as undefined/unsafe runtime behavior",
    },
    "task080": {
        "op": "MaxPool",
        "unused_output": "Values",
        "empty_output_probe": "checker rejects omission: required Single output",
        "decision": "no standard indices-only MaxPool form",
    },
    "task131": {
        "op": "TopK",
        "unused_output": "Values",
        "empty_output_probe": "checker rejects omission: required Single output",
        "decision": "also excluded by CenterCropPad and lookup lineage",
    },
    "task400": {
        "op": "MaxPool",
        "unused_output": "Values",
        "empty_output_probe": "checker rejects omission: required Single output",
        "decision": "no standard indices-only MaxPool form",
    },
    "replacement_analysis": {
        "Split_to_Slice": "not cost-beneficial: one singleton tensor is avoided, but Slice needs new int64 starts/ends/axes parameters and no reusable parameter bank exists",
        "Split_to_Gather": "rejected: adds lookup lineage and parameters",
        "MaxPool_indices_only": "no schema-safe standard operator replacement preserving the required indices semantics",
        "TopK_indices_only": "TopK Values is required; alternatives add forbidden lookup/semantic risk",
    },
}

result = {
    "lane": "A39",
    "authority": {
        "path": str(AUTHORITY.relative_to(ROOT)),
        "sha256": authority_sha,
        "unchanged": True,
        "score": 8000.46,
    },
    "decision": "NO_ADOPTABLE_CANDIDATE",
    "score_gain": 0.0,
    "scan": {
        "tasks_scanned": 400,
        "finding_tasks": len(finding_tasks),
        "single_output_dead_node_tasks": dead_node_tasks,
        "partial_multi_output_count": scan["partial_multi_output_count"],
        "candidate_count": scan["candidate_count"],
        "errors": scan["errors"],
    },
    "named_dead_nodes": named_summary,
    "multi_output_analysis": multi_output_summary,
    "validation": {
        "known_dual": "all four requested prune candidates rejected by runtime/session gate",
        "fresh_dual_5000": "not run: no candidate survived safety and known-runtime gates",
        "external_500": "not run: no candidate survived safety and known-runtime gates",
        "truthful_shapes": "no submission model changed",
    },
    "evidence": [
        "scan_build_manifest.json",
        "named_dead_prune_probes.json",
        "multi_output_omission_probes.json",
        "split_replacement_anatomy.json",
        "rejected_named_dead_prunes/",
        "rejected_multi_output_probes/",
        "../lane_b18/REPORT.md",
        "../../../scratch_codex_7991/lane32/REPORT.md",
    ],
}

(LANE / "A39_RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

report = f"""# A39 dead-node / dead-output audit

## Outcome

- Decision: **NO_ADOPTABLE_CANDIDATE**
- Authority score: **8000.46**
- Score gain: **+0.00**
- Authority SHA-256: `{authority_sha}` (unchanged)
- Scanned 400 models; found four single-output dead nodes and 46 partial multi-output sites.

## Requested dead nodes

| Task | Prune | Result | Safety lineage |
|---|---|---|---|
| 039 | `Equal keep_bg_equal` | 264/264 known runtime errors in both modes; Slice buffer-reuse mismatch | CenterCropPad |
| 089 | `ReduceMax keep_red_big` | default session creation fails; historical disabled probe fails all known cases | CenterCropPad |
| 122 | `GreaterOrEqual d_keep` | 266/266 known errors after pruning in both modes | CenterCropPad |
| 183 | `Min hold_u8` | 265/265 known runtime errors in both modes; Resize buffer-reuse mismatch | lookup/ScatterElements |

The apparently dead tensors are allocator/shape barriers. Removing them changes ORT buffer reuse and is therefore not a valid optimization.

## Partial multi-output nodes

- task019/task124 `Split`: replacing the unused variadic output with `""` passes ONNX checker but crashes ORT with SIGSEGV. This is rejected as undefined/unsafe behavior.
- task080/task400 `MaxPool`: the first Values output is schema-required even when only Indices is consumed.
- task131 `TopK`: Values is schema-required; the source also has forbidden CenterCropPad and lookup lineage.
- Replacing the two safe `Split` sites with individual `Slice` nodes is larger: it removes one singleton allocation but requires new int64 starts/ends/axes parameters, with no reusable initializer bank. `Gather` would add forbidden lookup lineage.

## Validation gate

No candidate survived the safety and known-runtime gates, so fresh dual 5000 and external 500 were correctly not run. No authority model or archive was modified.
"""
(LANE / "REPORT.md").write_text(report)

print(json.dumps({"result": "A39_RESULT.json", "authority_sha256": authority_sha, "decision": result["decision"]}))
