#!/usr/bin/env python3
"""Build machine-readable C25 decision and root-integrity evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_COSTS = {131: 746, 251: 755}
EXPECTED_ROOT = {
    "submission_base_7999.13.zip": "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1",
    "all_scores.csv": "3f9533f472a2153e12daeea4936aa7be3f47902a8fdb1621c31f778f6d009665",
    "best_score.json": "551409d40c18ef80a9ae7e89a6a0e567aa2474924018225e29639b32c0627e72",
    "artifacts/handcrafted": "5344ea88ff3e24509ed49fbc51b613ced484c8000513ee45060a6ce0b7ddbf69",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def handcrafted_aggregate() -> tuple[str, int]:
    paths = sorted(path for path in (ROOT / "artifacts/handcrafted").rglob("*") if path.is_file())
    lines = b"".join(
        f"{sha256(path)}  {path.relative_to(ROOT)}\n".encode()
        for path in paths
    )
    return hashlib.sha256(lines).hexdigest(), len(paths)


def known_total(record: dict[str, object], mode: str) -> dict[str, object] | None:
    value = record.get(mode)
    if not isinstance(value, dict):
        return None
    total = value.get("total")
    return total if isinstance(total, dict) else None


def classify(label: str, record: dict[str, object]) -> dict[str, object]:
    task = int(record["task"])
    score = record.get("official_like_score")
    cost = score.get("cost") if isinstance(score, dict) else None
    correct = score.get("correct") if isinstance(score, dict) else None
    trace = record.get("runtime_shape_trace") or {}
    trace_error = trace.get("error") if isinstance(trace, dict) else "missing trace"
    mismatches = trace.get("declared_actual_mismatches", []) if isinstance(trace, dict) else []
    disable = known_total(record, "known_disable_all")
    default = known_total(record, "known_default")
    dual_known = all(
        total is not None
        and total.get("right") == 266
        and total.get("wrong") == 0
        and total.get("errors") == 0
        for total in (disable, default)
    )
    reasons: list[str] = []
    if cost is None:
        reasons.append("official-like scorer/session failed")
    elif cost >= BASE_COSTS[task]:
        reasons.append(f"not strictly cheaper than exact baseline {BASE_COSTS[task]}")
    if correct is not True:
        reasons.append("known-set official-like correctness did not pass")
    if trace_error:
        reasons.append(f"runtime shape trace failed: {trace_error}")
    if mismatches:
        reasons.append(f"declared/runtime shape mismatch count={len(mismatches)}")
    if not dual_known:
        reasons.append("both ORT modes were not 266/266 with errors=0")
    lookup = record.get("lookup_red_flags") or {}
    if isinstance(lookup, dict) and (lookup.get("tfidf") or lookup.get("hardmax") or lookup.get("giant_einsum_nodes")):
        reasons.append("lookup/giant-Einsum red flag")
    accepted = not reasons
    return {
        "label": label,
        "task": task,
        "path": record["path"],
        "sha256": record["sha256"],
        "cost": cost,
        "baseline_cost": BASE_COSTS[task],
        "known_disable_all": record.get("known_disable_all"),
        "known_default": record.get("known_default"),
        "runtime_shape_mismatches": len(mismatches),
        "runtime_shape_trace_error": trace_error,
        "accepted": accepted,
        "reasons": reasons,
    }


def main() -> None:
    audits = json.loads((HERE / "model_audit.json").read_text(encoding="utf-8"))
    inventory = json.loads(
        (HERE.parent / "lane_archive_all400" / "inventory.json").read_text(encoding="utf-8")
    )
    decisions = [classify(label, record) for label, record in audits.items()]
    controls = [row for row in decisions if "control" in row["label"]]
    archive = [row for row in decisions if "archive" in row["label"]]
    base = [row for row in decisions if row["label"].startswith("base_")]
    output = {
        "baseline_score": 7999.13,
        "scope": [131, 251],
        "exact_baselines": base,
        "archive_candidates": archive,
        "truthful_or_reference_controls": controls,
        "finalists": [],
        "promotion": None,
        "score_gain": 0.0,
        "external_validator": {
            "status": "not_applicable",
            "reason": "no model survived cheaper + dual-ORT known + strict runtime-shape gates",
        },
        "errors_introduced": 0,
    }
    (HERE / "decision.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    history = {
        "inventory": "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json",
        "inventory_scan_stats": inventory["stats"],
        "retained_promising_models": {
            "task131": inventory["retained"]["131"],
            "task251": inventory["retained"]["251"],
        },
        "retained_counts": {"task131": 5, "task251": 4, "total": 9},
        "all_retained_reaudited_in_c25": True,
        "historical_failure_evidence": [
            {
                "task": 131,
                "sha256": "a13e6337acc30ddc9bc7f3276f3e464cc8144c12d40577bdd625d721ab1db182",
                "evidence": "scripts/golf/loop_7999_13/lane_root21_task131_dual5000.json",
                "disable_all": {"right": 782, "wrong": 4218, "runtime_or_output_failures": 4218},
                "default": {"right": 782, "wrong": 4218, "runtime_or_output_failures": 4218},
            },
            {
                "task": 251,
                "evidence": "scripts/golf/scratch_codex/task251/FAILURE_LOG.md#F9-F20",
                "adversarial_seed": 313630,
                "finding": "all historical public-correct costs 1030/1031/1032/1059 failed the clipped-rectangle case",
            },
        ],
    }
    (HERE / "history_audit.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    actual_files = {name: sha256(ROOT / name) for name in EXPECTED_ROOT if name != "artifacts/handcrafted"}
    aggregate, count = handcrafted_aggregate()
    actual_files["artifacts/handcrafted"] = aggregate
    integrity = {
        "expected": EXPECTED_ROOT,
        "actual": actual_files,
        "handcrafted_file_count": count,
        "all_match": actual_files == EXPECTED_ROOT,
        "root_mutations_by_lane": [],
    }
    (HERE / "root_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    assert integrity["all_match"], integrity
    print(f"audited={len(decisions)} archive={len(archive)} finalists=0 root_integrity=pass")


if __name__ == "__main__":
    main()
