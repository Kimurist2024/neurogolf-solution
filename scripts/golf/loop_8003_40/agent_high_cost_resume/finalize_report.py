#!/usr/bin/env python3
"""Finalize the high-cost sound-rebuild lane without merging candidates."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load(name: str) -> object:
    return json.loads((HERE / name).read_text())


def main() -> None:
    ranking = load("ranking_selection.json")
    refs = {row["task"]: row for row in load("rule_reference_fresh5000.json")}
    builds = load("task237_build_attempts.json")

    tasks = [
        {
            "task": 156,
            "classification": "Type A with global two-component extent comparison",
            "truth_rule": (
                "Find the two solid yellow rectangles; the smaller-area rectangle gets "
                "interior color 1 and the larger-area rectangle gets interior color 2; "
                "yellow borders remain unchanged."
            ),
            "reference_fresh5000": refs[156],
            "baseline": {"memory": 330, "params": 226, "cost": 556},
            "clean_floor": {
                "assessment": "AT_OR_BELOW_PRACTICAL_CLEAN_FLOOR",
                "reason": (
                    "A conventional 10x10 float color decode already costs 400 bytes before "
                    "component extent comparison and two interior masks. The incumbent performs "
                    "the classification in 330 bytes total memory and emits the free output "
                    "directly with a 180-parameter 2->10 QLinearConv. Replacing that final with "
                    "a 30x30 label/condition tensor alone costs at least 900 bytes."
                ),
            },
            "candidate": None,
            "verdict": "NO_CANDIDATE",
        },
        {
            "task": 237,
            "classification": "Type B bounded propagation (grid <=9)",
            "truth_rule": (
                "For each marker, paint its row from the marker to the right edge and carry "
                "the latest marker color down the last column."
            ),
            "reference_fresh5000": refs[237],
            "baseline": {"memory": 413, "params": 116, "cost": 529},
            "clean_floor": {
                "assessment": "AT_OR_BELOW_PRACTICAL_CLEAN_FLOOR",
                "reason": (
                    "A straightforward max-9 implementation needs a 9x9 float decode (324 bytes) "
                    "and at least one second full propagation grid (another 324 bytes), exceeding "
                    "the entire incumbent cost before parameters. The incumbent's packed scalar/"
                    "uint8 representation is already substantially below that floor."
                ),
            },
            "built_attempts": builds["attempts"],
            "early_validation": [
                {
                    "label": "remove_min",
                    "cost": 520,
                    "fresh_probe": {"right": 3, "wrong": 17, "total": 20},
                    "known_complete": False,
                    "verdict": "REJECT",
                    "reason": "Min clamps empty-row sentinel 9 to the active width; it is not redundant.",
                },
                {
                    "label": "shift_shrink",
                    "cost": 528,
                    "fresh_probe": {"right": 0, "wrong": 20, "total": 20},
                    "known_complete": False,
                    "verdict": "REJECT",
                    "reason": "The unshifted width is also consumed by Min; moving 15 only into column indices is invalid.",
                },
                {
                    "label": "combined",
                    "cost": 519,
                    "fresh_probe": {"right": 0, "wrong": 100, "total": 100},
                    "known_complete": False,
                    "verdict": "REJECT",
                    "reason": "Both required sentinel/width guards were removed.",
                },
            ],
            "candidate": None,
            "verdict": "NO_CANDIDATE",
        },
        {
            "task": 345,
            "classification": "Type B bounded path unroll (9 steps)",
            "truth_rule": (
                "From each red start on the bottom row, move upward; if gray blocks the cell "
                "above, move right instead; paint every visited cell red."
            ),
            "reference_fresh5000": refs[345],
            "baseline": {"memory": 248, "params": 141, "cost": 389},
            "clean_floor": {
                "assessment": "AT_OR_BELOW_PRACTICAL_CLEAN_FLOOR",
                "reason": (
                    "A direct nine-step 10x10 boolean path unroll requires at least 900 bytes of "
                    "step masks before decode. The incumbent represents all rows as scalar bitsets "
                    "and uses only 248 bytes of intermediate memory. Its 100-element shared row "
                    "kernel is dense under the official parameter counter; sparse_initializer is "
                    "a known grader-risk and is forbidden."
                ),
            },
            "candidate": None,
            "verdict": "NO_CANDIDATE",
        },
    ]

    attempts = [
        {"task": 156, "idea": "single encoded feature channel + grouped final conv", "result": "REJECT_DESIGN", "reason": "one affine channel cannot independently emit color1, color2, and the common yellow border"},
        {"task": 156, "idea": "split four active output channels then pad/permute", "result": "REJECT_FLOOR", "reason": "materializes at least 4x30x30=3600 bytes before the free output"},
        {"task": 156, "idea": "conventional float color decode and explicit masks", "result": "REJECT_FLOOR", "reason": "400-byte decode plus classification masks exceeds cost556"},
        {"task": 237, "idea": "remove generator-domain Min guard", "result": "REJECT_TEST", "reason": "3/20 fresh"},
        {"task": 237, "idea": "absorb Shrink bias into column indices", "result": "REJECT_TEST", "reason": "0/20 fresh"},
        {"task": 237, "idea": "combine both guard removals", "result": "REJECT_TEST", "reason": "0/100 fresh"},
        {"task": 237, "idea": "rank-3 factorization of the 10x8 packed kernel", "result": "REJECT_FLOOR", "reason": "dynamic reconstructed weight adds 320 bytes to save only 26 parameters"},
        {"task": 345, "idea": "float arithmetic to remove six scalar Cast outputs", "result": "REJECT_SEMANTICS", "reason": "BitwiseAnd on row bitsets cannot be replaced by Mul"},
        {"task": 345, "idea": "sparse Wpack initializer", "result": "REJECT_POLICY", "reason": "sparse_initializer is a known grader-error/risk lineage"},
    ]

    report = {
        "baseline": {
            "score": 8003.40,
            "zip": ranking["baseline"],
            "sha256": ranking["baseline_sha256"],
        },
        "ranking": {
            "range": ranking["range"],
            "count": ranking["ranked_count"],
            "selected": ranking["selected"],
            "source": "ranking_selection.json",
        },
        "selected_task_reports": tasks,
        "serious_attempts": attempts,
        "accepted": [],
        "projected_gain": 0.0,
        "verdict": "NO_CANDIDATE",
        "merge_performed": False,
        "protected_files_changed": False,
        "stop_reason": "All three incumbents are already below the practical clean structural floor; all cheaper concrete task237 probes failed immediately.",
    }
    (HERE / "FINAL_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    markdown = """# High-cost safe rebuild report (tasks 150-400)

## Outcome

- Ranked 251 baseline tasks using truthful runtime cost.
- Selected the top three sound Type A/B rebuilds after excluding private-zero, unsound, contaminated, other-lane, lookup, shape-cloak, Type C/D, and giant-Einsum tasks.
- Input-only rule references pass **5000/5000 fresh** for task156, task237, and task345.
- Safe lower-cost ONNX candidates: **0**.
- ZIP merge: **not performed**.
- Final verdict: **NO_CANDIDATE**.

| Task | Type | Baseline cost | Clean-floor conclusion |
|---:|---|---:|---|
| 156 | A + two-component extent comparison | 556 (330 memory + 226 params) | Conventional decode/masks exceed the incumbent; 30x30 label/condition alone is >=900 bytes. |
| 237 | B, bounded right/down propagation | 529 (413 + 116) | Honest 9x9 decode + one propagation grid is >=648 bytes before params. |
| 345 | B, nine-step obstacle path | 389 (248 + 141) | Direct boolean unroll is >=900 bytes; incumbent scalar bitsets are already below the clean floor. |

## Concrete task237 probes

- `remove_min`, cost 520: **REJECT**, fresh 3/20. Empty rows require the sentinel clamp.
- `shift_shrink`, cost 528: **REJECT**, fresh 0/20. The unshifted width is also required by `Min`.
- combined, cost 519: **REJECT**, fresh 0/100.

All three files pass full checker, strict shape inference/data propagation, and Conv-family bias-length checks, but fail correctness and are not candidates. Nine serious design/build attempts are recorded in `FINAL_REPORT.json`; further work stops at the structural floor.
"""
    (HERE / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"verdict": "NO_CANDIDATE", "accepted": 0, "attempts": len(attempts)}, indent=2))


if __name__ == "__main__":
    main()
