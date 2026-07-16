#!/usr/bin/env python3
"""Build true-rule and complete-history evidence for low44."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (303, 98, 395, 167, 289, 38, 262, 269)
RULES = {
    303: "Add 2 to a cell exactly when its row or column is all zero.",
    98: "Keep a cell exactly when its generator-defined clipped 3x3 neighborhood contains zero.",
    395: "Delete row index 3 and output 2 at positions where the remaining-row cell and aligned removed-row cell are both zero.",
    167: "Produce the fixed 3x3 color-5 permutation pattern selected by the distinct-character count modulo 5.",
    289: "Repeat every input cell equally in both dimensions by the generator's distinct-character-count scale.",
    38: "Produce a 1x5 prefix of ones from horizontal adjacent-one occurrences, then zeros.",
    262: "For each input row emit three copies of 3+(second-first)/5.",
    269: "Repeat every input cell equally in both dimensions by the generator's distinct-character-count scale.",
}


def normalize(value):
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def rule_row(task: int) -> dict[str, object]:
    source = ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"low44_task{task:03d}", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.loads((ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text())
    right = wrong = errors = 0
    split_counts = {}
    first_failure = None
    input_shapes = set()
    output_shapes = set()
    for split, examples in payload.items():
        if not isinstance(examples, list):
            continue
        for index, example in enumerate(examples):
            if not isinstance(example, dict) or "input" not in example:
                continue
            split_counts[split] = split_counts.get(split, 0) + 1
            input_shapes.add((len(example["input"]), len(example["input"][0])))
            output_shapes.add((len(example["output"]), len(example["output"][0])))
            try:
                actual = normalize(module.p(copy.deepcopy(example["input"])))
                if actual == example["output"]:
                    right += 1
                else:
                    wrong += 1
                    if first_failure is None:
                        first_failure = {"split": split, "index": index, "kind": "wrong", "actual": actual, "expected": example["output"]}
            except Exception as exc:
                errors += 1
                if first_failure is None:
                    first_failure = {"split": split, "index": index, "kind": "error", "error": f"{type(exc).__name__}: {exc}"}
    total = right + wrong + errors
    return {
        "task": task,
        "source": str(source.relative_to(ROOT)),
        "summary": RULES[task],
        "known": {"right": right, "wrong": wrong, "errors": errors, "total": total, "perfect": right == total and total > 0, "first_failure": first_failure},
        "split_counts": split_counts,
        "input_shapes": [list(x) for x in sorted(input_shapes)],
        "output_shapes": [list(x) for x in sorted(output_shapes)],
    }


def main() -> None:
    rules = [rule_row(task) for task in TARGETS]
    (HERE / "true_rule_audit.json").write_text(json.dumps({"targets_completed": len(rules), "all_perfect": all(row["known"]["perfect"] for row in rules), "rows": rules}, indent=2) + "\n")

    archive = json.loads((ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text())
    harvest = json.loads((ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json").read_text())
    exact = json.loads((ROOT / "scripts/golf/loop_8004_42_plus20/agent_exact_wave2/static_scan.json").read_text())
    einsum = json.loads((ROOT / "scripts/golf/loop_8004_42_plus20/root_exact_einsum25/scan_report.json").read_text())
    opportunities = []
    for family, rows in exact.get("opportunities", {}).items():
        opportunities.extend({"family": family, **row} for row in rows if row.get("task") in TARGETS)
    exact_candidates = [row for row in exact.get("candidates", []) if row.get("task") in TARGETS]
    einsum_hits = []
    for family in ("initializer_dedup", "outer_fusion", "sign_absorption"):
        einsum_hits.extend({"family": family, **row} for row in einsum.get(family, []) if row.get("task") in TARGETS)
    history = {
        "all400_archive": {
            "baseline": archive["base"],
            "stats": archive["stats"],
            "retained_strict_lower": {str(task): archive["retained"].get(str(task), []) for task in TARGETS},
        },
        "focused_harvest": {
            "inventory": harvest["inventory"],
            "target_rows": {str(task): [row for row in harvest["rows"] if row.get("task") == task] for task in TARGETS},
        },
        "exact_wave2": {
            "summary": exact["summary"],
            "target_opportunities": opportunities,
            "target_candidates": exact_candidates,
        },
        "exact_einsum_all400": {
            "task_count": einsum.get("task_count"),
            "target_hits": einsum_hits,
        },
        "conclusion": "No archived strict-lower candidate exists for any target. The only exact opportunities are Identity removals for task262/269/289, all rejected by full checker/strict inference due inherited false shape contracts.",
    }
    (HERE / "history_audit.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"rules": {str(row["task"]): row["known"] for row in rules}, "strict_lower_counts": {str(task): len(archive["retained"].get(str(task), [])) for task in TARGETS}, "exact_hits": opportunities}, indent=2))


if __name__ == "__main__":
    main()
