#!/usr/bin/env python3
"""Profile the full retained/later strict-lower frontier for six private tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.loop_8004_42_plus20.audit_retained_group import profile  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import (  # noqa: E402
    run_known,
    structure,
)


TARGETS = (134, 202, 219, 271, 343, 396)
EXTRAS = [
    {
        "task": 219,
        "path": "scripts/golf/loop_7999_13/lane_b32/task219_b32_winner.onnx",
        "source": "later exact-current algebraic reduction; not a true-rule compiler",
    },
    {
        "task": 396,
        "path": "scripts/golf/scratch_codex/task396/cand_rule_k2.onnx",
        "source": "later true-rule reconstruction attempt k2",
    },
    {
        "task": 396,
        "path": "scripts/golf/scratch_codex/task396/cand_rule_k3.onnx",
        "source": "later true-rule reconstruction attempt k3",
    },
    {
        "task": 396,
        "path": "scripts/golf/scratch_codex/task396/cand_rule_k4.onnx",
        "source": "later true-rule reconstruction attempt k4",
    },
    {
        "task": 396,
        "path": "scripts/golf/scratch_codex/task396/cand_rule_k4_occupancy.onnx",
        "source": "later true-rule reconstruction attempt k4 occupancy",
    },
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def perfect(row: dict[str, object]) -> bool:
    return bool(row.get("total")) and row.get("right") == row.get("total") and row.get("errors") == 0


def main() -> None:
    full = json.loads((ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text())
    loose = json.loads((ROOT / "scripts/golf/loop_7999_13/lane_archive_loose_sweep/inventory.json").read_text())
    previous = json.loads((HERE / "retained_audit.json").read_text())
    baseline_cost = {
        int(row["task"]): int(row["actual_cost"]["cost"])
        for row in previous["baseline_rows"]
    }
    records: list[dict[str, object]] = []
    for task in TARGETS:
        loose_rows = loose.get("retained", {}).get(str(task), []) or []
        full_rows = full.get("retained", {}).get(str(task), []) or []
        chosen = loose_rows if len(loose_rows) >= len(full_rows) else full_rows
        for row in chosen:
            records.append(
                {
                    "task": task,
                    "path": row["path"],
                    "archive_reported_static_cost": row.get("static_cost"),
                    "archive_sources": row.get("sources", []),
                    "inventory": "lane_archive_loose_sweep" if chosen is loose_rows else "lane_archive_all400",
                }
            )
    records.extend(EXTRAS)

    unique: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()
    for record in records:
        path = ROOT / str(record["path"])
        digest = sha(path.read_bytes())
        key = (int(record["task"]), digest)
        if key in seen:
            continue
        seen.add(key)
        record = dict(record)
        record["sha256"] = digest
        unique.append(record)

    rows = []
    for index, record in enumerate(unique, start=1):
        task = int(record["task"])
        path = ROOT / str(record["path"])
        data = path.read_bytes()
        model = onnx.load_model_from_string(data)
        actual = profile(data, path.name)
        disable = run_known(copy.deepcopy(model), task, True)
        default = run_known(copy.deepcopy(model), task, False) if perfect(disable) else {"not_run": "disable_all_not_perfect"}
        structural = structure(copy.deepcopy(model), task)
        einsum_inputs = [len(node.input) for node in model.graph.node if node.op_type == "Einsum"]
        row = {
            **record,
            "baseline_actual_cost": baseline_cost[task],
            "actual_cost": actual,
            "strictly_lower_actual": actual.get("cost") is not None and int(actual["cost"]) < baseline_cost[task],
            "known_disable_all": disable,
            "known_default": default,
            "known_dual_perfect": perfect(disable) and perfect(default),
            "structure": structural,
            "derived_flags": {
                "einsum_input_counts": einsum_inputs,
                "giant_einsum_ge8": any(value >= 8 for value in einsum_inputs),
                "lookup_or_scatter": structural.get("lookup_or_scatter", []),
                "shape_cloak": bool(structural.get("runtime_shapes", {}).get("shape_cloak")),
                "structural_error_count": len(structural.get("errors", [])),
            },
        }
        rows.append(row)
        print(
            f"{index:02d}/{len(unique)} task{task:03d} cost={actual.get('cost')} "
            f"known={disable.get('right')}/{disable.get('total')} "
            f"lookup={len(row['derived_flags']['lookup_or_scatter'])} "
            f"cloak={row['derived_flags']['shape_cloak']}",
            flush=True,
        )

    counts = []
    for task in TARGETS:
        selected = [row for row in rows if row["task"] == task]
        counts.append(
            {
                "task": task,
                "baseline_actual_cost": baseline_cost[task],
                "unique_candidate_files": len(selected),
                "strictly_lower_actual": sum(bool(row["strictly_lower_actual"]) for row in selected),
                "known_dual_perfect": sum(bool(row["known_dual_perfect"]) for row in selected),
                "lookup_or_scatter": sum(bool(row["derived_flags"]["lookup_or_scatter"]) for row in selected),
                "shape_cloak": sum(bool(row["derived_flags"]["shape_cloak"]) for row in selected),
                "giant_einsum_ge8": sum(bool(row["derived_flags"]["giant_einsum_ge8"]) for row in selected),
            }
        )
    result = {
        "scope": {
            "archive_policy": "larger per-task unique retained frontier from lane_archive_loose_sweep or lane_archive_all400",
            "later_candidates": EXTRAS,
            "deduplication": "task + SHA-256",
            "baseline": "submission_base_8005.17.zip",
            "baseline_sha256": "c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04",
            "baseline_note": "8005.16 -> 8005.17 changes task226 only; all six audited members are byte-identical",
        },
        "targets": list(TARGETS),
        "counts": counts,
        "aggregate": {
            "unique_candidate_files": len(rows),
            "strictly_lower_actual": sum(bool(row["strictly_lower_actual"]) for row in rows),
            "known_dual_perfect": sum(bool(row["known_dual_perfect"]) for row in rows),
            "task_histogram": dict(Counter(str(row["task"]) for row in rows)),
        },
        "rows": rows,
    }
    (HERE / "extended_candidate_audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
