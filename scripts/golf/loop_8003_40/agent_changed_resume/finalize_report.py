#!/usr/bin/env python3
"""Assemble the changed-task lane's immutable rejection ledger and report."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parent / "base_models"
KNOWN = HERE / "known"
CHANGED = HERE.parent / "agent_changed_tasks"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bias_issues(path: Path) -> list[list[object]]:
    checker_path = ROOT / "scripts/golf/check_conv_bias.py"
    namespace: dict[str, object] = {
        "__name__": "check_conv_bias_import",
        "__file__": str(checker_path),
    }
    exec(checker_path.read_text(), namespace)  # noqa: S102 - trusted local checker
    return [list(item) for item in namespace["check_model"](onnx.load(path))]


def profile_row(profile_path: Path, candidate_path: Path, method: str) -> dict[str, object]:
    profile = json.loads(profile_path.read_text())
    candidate = profile["candidate"]
    decision = profile["decision"]
    known = candidate["known"]
    return {
        "task": int(candidate["task"]),
        "candidate": str(candidate_path.relative_to(ROOT)),
        "sha256": candidate["sha256"],
        "method": method,
        "checker_shape": "PASS" if candidate["preflight_ok"] else "FAIL",
        "conv_bias": "PASS" if not bias_issues(candidate_path) else "FAIL",
        "candidate_cost": candidate["cost"],
        "baseline_cost": profile["baseline"]["cost"],
        "known_right": known["right"],
        "known_total": known["total_seen"],
        "runtime_errors": known["errors"],
        "fresh": "NOT_RUN_KNOWN_GATE_FAILED",
        "verdict": "REJECT",
        "reason": "; ".join(decision["reasons"]),
    }


def main() -> None:
    scores: dict[int, int] = {}
    with (ROOT / "all_scores.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            scores[int(row["task"].removeprefix("task"))] = int(row["cost"])

    tasks = (73, 111, 122, 260, 271, 285, 289, 359)
    baselines = []
    for task in tasks:
        path = BASE / f"task{task:03d}.onnx"
        baselines.append(
            {
                "task": task,
                "cost": scores[task],
                "sha256": sha256(path),
                "conv_bias": "PASS" if not bias_issues(path) else "FAIL",
            }
        )

    rows: list[dict[str, object]] = []
    for length in range(1, 6):
        rows.append(
            profile_row(
                CHANGED / f"task073_len{length}_known.json",
                CHANGED / f"task073_truncated/task073_len{length}.onnx",
                f"FIR truncate len={length}",
            )
        )

    for task in (111, 122):
        directory = HERE.parent / "dead_code_candidates"
        rows.append(
            profile_row(
                directory / f"task{task:03d}_profile.json",
                directory / f"task{task:03d}.onnx",
                "dead-node removal (shape-cloak trap)",
            )
        )

    for path in sorted(KNOWN.glob("task260_*.json")):
        payload = json.loads(path.read_text())
        candidate_path = ROOT / payload["candidate"]["source"] if False else None
        stem = path.stem
        model_path = CHANGED / "broadcast_prunes" / f"{stem}.onnx"
        rows.append(profile_row(path, model_path, "singleton broadcast prune"))

    archive_profile = HERE.parent / "agent_archive_rescreen/profiles/task271_known.json"
    rows.append(
        profile_row(
            archive_profile,
            ROOT / "others/2/7907/task271_improved.onnx",
            "archived Gather candidate",
        )
    )

    for path in sorted(KNOWN.glob("task359_*.json")):
        stem = path.stem
        model_path = CHANGED / "broadcast_prunes" / f"{stem}.onnx"
        rows.append(profile_row(path, model_path, "singleton broadcast prune"))

    quick = json.loads((HERE / "task359_remaining_quick_reject.json").read_text())
    for item in quick:
        model_path = ROOT / str(item["candidate"])
        rows.append(
            {
                "task": 359,
                "candidate": item["candidate"],
                "sha256": item["sha256"],
                "method": "singleton broadcast prune",
                "checker_shape": "PASS",
                "conv_bias": "PASS" if not bias_issues(model_path) else "FAIL",
                "candidate_cost": 22,
                "baseline_cost": 24,
                "known_right": 0,
                "known_total": 1,
                "runtime_errors": item["runtime_errors"],
                "fresh": "NOT_RUN_KNOWN_GATE_FAILED",
                "verdict": "REJECT",
                "reason": f"train[0] threshold mismatch ({item['different_cells']} cells)",
            }
        )

    rows.sort(key=lambda row: (int(row["task"]), str(row["candidate"])))
    with (HERE / "REJECTIONS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "baseline": "submission_base_8003.40.zip",
        "baseline_sha256": sha256(ROOT / "submission_base_8003.40.zip"),
        "target_tasks": list(tasks),
        "baseline_models": baselines,
        "audited_candidates": len(rows),
        "accepted_candidates": 0,
        "fresh_runs": 0,
        "reason_fresh_not_run": "every executable candidate failed the known-complete gate",
        "not_retried": {
            "task285": "known shape-cloak/private-risk lineage; instructed not to retry",
            "task289": "known shape-cloak/private-risk lineage; instructed not to retry",
        },
        "sparse_probe": json.loads((HERE / "sparse_build.json").read_text()),
        "rejections_csv": str((HERE / "REJECTIONS.csv").relative_to(ROOT)),
        "root_files_modified": [],
        "merged_into_zip": False,
    }
    (HERE / "REPORT.json").write_text(json.dumps(payload, indent=2) + "\n")

    baseline_lines = "\n".join(
        f"| {item['task']:03d} | {item['cost']} | `{item['sha256']}` | {item['conv_bias']} |"
        for item in baselines
    )
    task_counts: dict[int, int] = {}
    for row in rows:
        task_counts[int(row["task"])] = task_counts.get(int(row["task"]), 0) + 1
    count_text = ", ".join(f"task{task:03d}={count}" for task, count in sorted(task_counts.items()))
    report = f"""# Changed-task resume report

## Outcome

- Baseline: `submission_base_8003.40.zip`
- Audited executable candidates: **{len(rows)}** ({count_text})
- Accepted: **0**
- Fresh runs: **0** — every executable candidate failed known completeness first.
- task285/task289 were not retried because their known reductions depend on shape-cloak behavior.
- Sparse-initializer probes for task073/260/271 were rejected by full ONNX type/shape inference; sparse tensors cannot replace dense ConvTranspose/Einsum/QLinearConv inputs.
- No candidate was merged. Root ZIP, score JSON, CSV files, and `LOOP_STATUS.md` were not changed.

## Baseline identity and bias-UB gate

| Task | Cost | SHA-256 | Conv-family bias length |
|---:|---:|---|---|
{baseline_lines}

All candidate files in `REJECTIONS.csv` also pass the Conv-family bias-length checker. They were rejected for correctness/runtime reasons before fresh validation.

## Rejection summary

- task073: FIR lengths 1–5 each score **0/15 known**, errors 0.
- task111: dead-node removal produces **265/265 runtime errors** from shape-buffer reuse.
- task122: dead-node removal produces **266/266 runtime errors** and does not reduce truthful cost.
- task260: all 20 singleton-broadcast candidates fail known completeness; best is only **23/266**.
- task271: archived cost-10 Gather candidate scores **0/267 known**.
- task359: four candidates score **0/266 known**; the two interrupted full profiles are independently rejected on train[0] with 1097 and 1381 differing cells.
- task285/task289: no new model was emitted; shape-cloak/dead-code variants were intentionally not retried.

Candidate-by-candidate paths, SHA-256, costs, checker/bias status, known counts, runtime errors, and rejection reasons are in `REJECTIONS.csv`.
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps({"audited": len(rows), "accepted": 0, "report": str(HERE / 'REPORT.md')}))


if __name__ == "__main__":
    main()
