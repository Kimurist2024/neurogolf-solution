#!/usr/bin/env python3
"""Assemble the final eight-task decision and human-readable report."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def known_summary(entry: dict) -> dict:
    return {
        mode: {
            key: value
            for key, value in row.items()
            if key in {"total", "right", "errors", "rate", "session_error"}
        }
        for mode, row in entry["known"].items()
    }


def main() -> int:
    baseline = json.loads((HERE / "baseline_audit.json").read_text())
    fresh = json.loads((HERE / "fresh_audit.json").read_text())
    root089 = json.loads((HERE / "task089_root_candidate_audit.json").read_text())
    diagnostics = json.loads((HERE / "diagnostic_build.json").read_text())
    reasons = {
        13: [
            "51-input terminal Einsum violates the no-giant-contraction gate",
            "the truthful spec control costs 7884, above the cost-638 incumbent",
        ],
        18: [
            "strict data-propagating shape inference fails",
            "default ORT session creation fails",
            "61 declared/runtime shape mismatches",
            "the clean truthful rebuild costs 10857, above 4753",
        ],
        54: [
            "default ORT session creation fails",
            "40 declared/runtime shape mismatches",
            "the dual/fresh-perfect truthful control costs 49618, above 2258",
        ],
        80: [
            "no strictly cheaper candidate survived exact scanning",
            "no Identity, duplicate producer, unused initializer, or further safe scalar-broadcast shave exists",
            "the prior cost-3073 ablations are now above the cost-3050 incumbent and fail stored gold",
        ],
        89: [
            "default ORT session creation fails",
            "49 declared/runtime shape mismatches",
            "the truthful spec-derived control costs 2620, above 1349",
            "the new cost-1180 root lead crashes all 267 known cases under ORT_DISABLE_ALL",
        ],
        96: [
            "default ORT session creation fails",
            "6 declared/runtime shape mismatches",
            "the truthful repair costs 2000, above 1128",
            "removing the only Identity exposes the shape cloak and prevents checker/ORT initialization",
        ],
        101: [
            "private-zero lineage requires 100% fresh correctness",
            "fresh correctness is 994/1000 and 987/1000 in both ORT modes",
            "the decoded-rule sound control costs 7264, above 5655",
        ],
        131: [
            "18 declared/runtime shape mismatches",
            "the independent spec rebuild costs 1521, above 691",
        ],
    }
    sound_controls = {
        13: {"cost": 7884, "path": "scripts/golf/scratch_claude/task013/cand.onnx"},
        18: {"cost": 10857, "path": "scripts/golf/scratch_codex/task018/tile2x3_k22_allmode_clean.onnx"},
        54: {"cost": 49618, "path": "scripts/golf/scratch_codex/task054/task054_vector.onnx"},
        80: {"cost": 3050, "path": "8005.16 incumbent (spec-derived floor)"},
        89: {"cost": 2620, "sha256": "c97fff30f5fa41cf8345791fbcd78b6ad0c0af4e6b25d9aed38258b952b6a683"},
        96: {"cost": 2000, "path": "agent_policy90_repairs7/candidates/task096_default_safe.onnx"},
        101: {"cost": 7264, "path": "scripts/golf/scratch_codex_7994/task101_sound/sound_7264.onnx"},
        131: {"cost": 1521, "path": "scripts/golf/scratch_claude/task131/cand_tie.onnx"},
    }
    rows = []
    for task in (13, 18, 54, 80, 89, 96, 101, 131):
        entry = baseline[f"task{task:03d}"]
        row = {
            "task": task,
            "baseline_sha256": entry["sha256"],
            "baseline_cost": entry["measured"]["cost"],
            "known_dual": known_summary(entry),
            "structure": {
                "checker_full": entry["checker_full"],
                "strict_data_prop": entry["strict_data_prop"],
                "static_positive": entry["static_positive"],
                "runtime_shapes_truthful": entry["runtime_shapes"]["truthful"],
                "runtime_shape_mismatches": entry["runtime_shapes"].get("mismatch_count"),
                "conv_bias_findings": len(entry["conv_bias_findings"]),
                "standard_domains": entry["standard_domains"],
                "max_einsum_inputs": entry["max_einsum_inputs"],
                "lookup_ops": entry["lookup_ops"],
            },
            "sound_control": sound_controls[task],
            "decision": "REJECT_NO_SAFE_IMPROVEMENT",
            "reasons": reasons[task],
        }
        if task in (80, 101):
            row["fresh_dual"] = fresh[f"task{task:03d}"]
        rows.append(row)
    result = {
        "baseline": {
            "zip": "submission_base_8005.16.zip",
            "zip_sha256": "73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00",
            "lb_score": 8005.16,
        },
        "policy": {
            "normal": "lower cost; complete known 100%; two independent fresh seeds each >=90%; dual ORT runtime0; strict/data_prop; truthful runtime shapes; UB0; no lookup/cloak/giant",
            "private_zero": "decoded true rule or exact LB-white lineage plus known/fresh 100% in both ORT modes; runtime0; strict/truthful/UB0",
        },
        "tasks": [13, 18, 54, 80, 89, 96, 101, 131],
        "rows": rows,
        "new_root_task089_candidate": root089,
        "diagnostic_task096_identity_prune": diagnostics,
        "accepted": [],
        "accepted_count": 0,
        "aggregate_projected_gain": 0.0,
        "final_verdict": "NO_SAFE_CANDIDATE",
        "protected_files_modified": False,
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    table = []
    for row in rows:
        task = row["task"]
        structure = row["structure"]
        fresh_note = "-"
        if task == 80:
            fresh_note = "5000/5000 ×2 seeds ×2 ORT"
        elif task == 101:
            fresh_note = "994/1000, 987/1000 ×2 ORT"
        blockers = "; ".join(row["reasons"][:2])
        table.append(
            f"| {task:03d} | {row['baseline_cost']} | {structure['runtime_shapes_truthful']} | "
            f"{fresh_note} | REJECT | {blockers} |"
        )
    report = f"""# 8005.16 rebase-new21 audit — 8 tasks

## Outcome

採用候補は **0件**、このレーンの加点は **+0.000000** です。`submission_base_8005.16.zip` と root の提出物・CSV・`best_score.json` は変更していません。

| task | incumbent cost | truthful runtime shapes | fresh | decision | main blocker |
|---:|---:|:---:|:---|:---:|:---|
{chr(10).join(table)}

## Important independent re-audit: task089 root lead

- Candidate: `scripts/golf/loop_8004_42_plus20/root_exact_noop26/task089.onnx`
- SHA-256: `{root089['candidate_sha256']}`
- Apparent cost: `{root089['baseline_cost']} -> {root089['candidate_cost']}` (`+{root089['projected_gain_if_valid']:.12f}` if valid)
- Actual gate: **REJECT**. ORT_DISABLE_ALL gives candidate correct `0/267` with runtime errors `267/267`; default ORT session creation fails. Raw equality to the LB incumbent is `0/267`. Runtime-shape contradictions: `{root089['runtime_shapes']['mismatch_count']}`. UB findings: `0`.
- Fresh was deliberately not run after the mandatory known/runtime/truthful gates failed.

## Only structurally healthy incumbent: task080

task080 is checker/strict/data_prop/static/truthful/UB0 clean and remains spec-derived. It passed `5000/5000` on each of two independent fresh seeds in both ORT modes (20,000 executions total), with runtime errors 0 and dual raw equality 100%. However, exact scan found no cheaper graph. The closest prior ablations cost 3073 (already above 3050) and fail stored gold.

## Other decisive blockers

- task013 is shape-truthful and known-perfect but terminates in a **51-input Einsum**, violating the explicit no-giant-contraction gate. The non-giant spec control costs 7884 versus 638.
- task018/054/089/096/131 have declared/runtime shape contradictions (61/40/49/6/18 respectively); several also fail default ORT. Truthful controls are all more expensive.
- task101 is structurally clean but is private-zero lineage. It scored only 994/1000 and 987/1000 on independent fresh seeds in both ORT modes, below the required 100%; the sound decoded-rule control costs 7264 versus 5655.
- task096's only apparent exact Identity prune is not valid: it exposes the shape carrier and then fails checker/ORT initialization.

## Evidence

- `baseline_audit.json`: complete known dual-ORT, cost, strict/data_prop, runtime-shape and UB audit for all 8 incumbents.
- `fresh_audit.json`: task080 and task101 independent fresh dual-ORT runs.
- `task089_root_candidate_audit.json`: the cost-1180 root lead rejection.
- `result.json`: machine-readable final decisions and exact SHA values.

Final decision: **do not merge any model from this lane**.
"""
    (HERE / "REPORT.md").write_text(report)
    print("wrote result.json and REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
