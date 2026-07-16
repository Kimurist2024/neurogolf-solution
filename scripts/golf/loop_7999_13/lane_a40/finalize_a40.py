#!/usr/bin/env python3
"""Finalize task396 SOUND/cost audit without modifying the authority ZIP."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8000.46.zip"
AUTHORITY_SHA = "74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534"
POOL = ROOT / "scripts/golf/scratch_codex/task396"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert digest(AUTHORITY) == AUTHORITY_SHA
with zipfile.ZipFile(AUTHORITY) as archive:
    authority_bytes = archive.read("task396.onnx")
authority_task_sha = hashlib.sha256(authority_bytes).hexdigest()
assert authority_task_sha == "ce0bd7c49e11cbde341756993a71618c5c0bf8e086de6caf56ad93e8588e1d94"

study = json.loads((HERE / "row_score_study.json").read_text())
control_fresh = json.loads(
    (ROOT / "scripts/golf/loop_7999_13/lane_b17/task396_control_fresh5000.json").read_text()
)

models = {
    "authority": ROOT / "/tmp/a40_task396_authority.onnx",
    "rule_k2": POOL / "cand_rule_k2.onnx",
    "rule_k3": POOL / "cand_rule_k3.onnx",
    "rule_k4": POOL / "cand_rule_k4.onnx",
    "rule_k4_occupancy": POOL / "cand_rule_k4_occupancy.onnx",
    "rule_k5": POOL / "cand_rule_k5.onnx",
    "rule_k6": POOL / "cand_rule_k6.onnx",
    "rule_k7": POOL / "cand_rule_k7.onnx",
    "rule_k8": POOL / "cand_rule_k8.onnx",
    "corner_micro": POOL / "agent_corner_micro.onnx",
    "spec_runs": POOL / "cand_spec_runs.onnx",
}
# The authority member was only extracted to /tmp for read-only inspection;
# keep the durable result independent of that temporary file.
models.pop("authority")

structural = {}
for label, path in models.items():
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    structural[label] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "nodes": len(model.graph.node),
        "ops": sorted({node.op_type for node in model.graph.node}),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "functions": len(model.functions),
    }

result = {
    "lane": "A40",
    "task": 396,
    "authority": {
        "zip": "submission_base_8000.46.zip",
        "zip_sha256": AUTHORITY_SHA,
        "task396_sha256": authority_task_sha,
        "task396_cost": 1019,
        "unchanged": True,
    },
    "rule": {
        "classification": "global object selection plus data-dependent crop",
        "summary": (
            "Find the uniquely widest/tallest hollow rectangle, crop it, and paint "
            "every nonzero crop cell with the other nonzero color."
        ),
        "reference_visible": "all 266 exact",
        "reference_fresh": {"right": 5000, "wrong": 0, "seed": 404039600},
    },
    "decision": "NO_ADOPTABLE_CANDIDATE",
    "score_gain": 0.0,
    "reason": (
        "No candidate is simultaneously cost < 1019, generator-SOUND, known exact, "
        "fresh exact, truthful-shape, and free of forbidden lookup/cloak/UB lineage."
    ),
    "candidate_table": [
        {"model": "authority", "cost": 1019, "known": "266/266", "fresh": "4954/5000 dual", "decision": "retain LB-white authority; not generator-SOUND"},
        {"model": "rule_k2", "cost": 875, "known": "fail", "fresh": "not eligible", "decision": "reject"},
        {"model": "rule_k3", "cost": 939, "known": "fail", "fresh": "not eligible", "decision": "reject"},
        {"model": "rule_k4", "cost": 1003, "known": "fail", "fresh": "not eligible", "decision": "reject"},
        {"model": "rule_k4_occupancy", "cost": 1014, "known": "266/266", "fresh": "9906/10000", "decision": "reject: row-reduction heuristic"},
        {"model": "rule_k5", "cost": 1067, "known": "fail", "fresh": "not eligible", "decision": "reject"},
        {"model": "rule_k6", "cost": 1131, "known": "fail", "fresh": "not eligible", "decision": "reject"},
        {"model": "rule_k7", "cost": 1195, "known": "266/266", "fresh": "1000/1000 historical", "decision": "reject: over cost"},
        {"model": "corner_micro", "cost": 1245, "known": "266/266", "fresh": "5000/5000 dual (also historical 10000/10000)", "decision": "reject: over cost and lookup-like operator lineage"},
        {"model": "spec_runs", "cost": 16457, "known": "266/266", "fresh": "1000/1000 historical", "decision": "reject: over cost and lookup operators"},
    ],
    "cheap_row_score_study": {
        "cases": study["count"],
        "seed": study["seed"],
        "failures_requiring_both_true_border_rows_in_top4": {
            key: study["stats"].get(f"{key}_k4_fail", 0)
            for key in ("laplace", "absdiff", "peak", "minedge", "count")
        },
        "conclusion": "No tested one-linear-row-score replacement can soundly replace the exact corner/run stage.",
    },
    "control_fresh_evidence": control_fresh,
    "structural": structural,
    "evidence": [
        "row_score_study.json",
        "analyze_row_scores.py",
        "../../lane_b17/task396_control_fresh5000.json",
        "../../../scratch_codex/task396/REPORT.md",
        "../../../scratch_codex/task396/FAILURE_LOG.md",
    ],
}

(HERE / "A40_RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

report = f"""# A40 task396 SOUND rebuild audit

## Outcome

- Decision: **NO_ADOPTABLE_CANDIDATE**
- Required threshold: **cost < 1019**
- Score gain: **+0.00**
- Authority task396 SHA-256: `{authority_task_sha}`
- Authority ZIP SHA-256: `{AUTHORITY_SHA}` (unchanged)

The generator rule was independently replayed on all 266 visible cases and
5,000 fresh cases: find the uniquely widest/tallest hollow rectangle, crop it,
and repaint every nonzero crop cell with the other nonzero color.

## Cost/correctness boundary

| model | cost | known | fresh | decision |
|---|---:|---:|---:|---|
| authority | 1019 | 266/266 | 4954/5000 dual | retain current LB-white member |
| rule k2 | 875 | fail | — | reject |
| rule k3 | 939 | fail | — | reject |
| rule k4 | 1003 | fail | — | reject |
| rule k4 occupancy | 1014 | 266/266 | 9906/10000 | reject; heuristic is not SOUND |
| rule k5 | 1067 | fail | — | reject |
| rule k6 | 1131 | fail | — | reject |
| rule k7 | 1195 | 266/266 | 1000/1000 | over threshold |
| corner micro | 1245 | 266/266 | 5000/5000 dual | over threshold; lookup-like ops |
| full spec runs | 16457 | 266/266 | 1000/1000 | over threshold; lookup ops |

The previous cheapest generator-SOUND control remains cost 1245.  Its exact
corner stage must retain all possible top-left rows before width selection;
reducing to 2–6 high-count rows is not generator-entailed.

## New low-cost decomposition check

I tested five one-linear-row-score replacements over {study['count']:,} fresh
generator cases.  Even retaining the top four rows, failures were:

- Laplacian score: {study['stats']['laplace_k4_fail']}
- absolute difference: {study['stats']['absdiff_k4_fail']}
- peak score: {study['stats']['peak_k4_fail']}
- minimum-edge score: {study['stats']['minedge_k4_fail']}
- plain count: {study['stats']['count_k4_fail']}

Projection overlaps from the other one or two rectangles can outrank either
border of the largest rectangle.  Thus a cheap linear score cannot replace the
nonlinear same-color corner/run test soundly.

No candidate survived the cost + SOUND gates, so no new known-dual/fresh-dual
5000/external-500 promotion run was warranted.  The shared ZIP was not changed.
"""
(HERE / "REPORT.md").write_text(report)
print(json.dumps({"decision": result["decision"], "score_gain": 0.0, "authority_unchanged": True}))
