"""Report generation for the NeuroGolf optimization pipeline.

Extracted from ``optimize_submission.py`` to keep that module under the
800-line budget. Builds the per-run markdown + JSON report (including the
proposal-002 residual-pass accounting) and the submission zip. Pipeline-level
config (output dirs, skip lists, G1 task set, margin, pass-set labels) is passed
in via ``ReportConfig`` so this module has no back-dependency on the driver.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib.pipeline_types import CandidateResult, TaskResult


@dataclass(frozen=True)
class ReportConfig:
    """Static configuration the report builder needs from the driver."""

    reports_dir: Path
    optimized_dir: Path
    submission_zip: Path
    skip_tasks_s3: list[int]
    g1_fp16_tasks: list[int]
    g2_margin: float
    passes_001: str
    passes_002: str


def next_run_number(reports_dir: Path) -> int:
    """Next free integer for ``run-NNN`` report filenames (deterministic)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    existing: list[int] = []
    for path in reports_dir.glob("run-*.md"):
        stem = path.stem.replace("run-", "")
        if stem.isdigit():
            existing.append(int(stem))
    return (max(existing) + 1) if existing else 1


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _candidate_to_json(cand: CandidateResult | None) -> dict[str, Any]:
    if cand is None:
        return {"present": False}
    return {
        "present": True,
        "source": cand.source,
        "correct": cand.correct,
        "memory": cand.memory,
        "params": cand.params,
        "cost": cand.cost,
        "score": cand.score,
        "is_baseline": cand.is_baseline,
        "note": cand.note,
    }


def build_reports(
    results: list[TaskResult],
    sources: list[str],
    run_number: int,
    passes: str,
    cfg: ReportConfig,
) -> tuple[str, dict[str, Any]]:
    """Build the markdown report text and machine-readable JSON payload."""
    rows: list[str] = []
    json_tasks: list[dict[str, Any]] = []
    warnings: list[str] = []

    total_score_before = 0.0
    total_score_after = 0.0
    # Realized score gain attributable to each 002 pass (summed over tasks).
    p002_gain: dict[str, float] = {"G3": 0.0, "G2": 0.0, "G1": 0.0}

    for res in results:
        baseline = res.baseline
        chosen = res.chosen

        b_params = baseline.params if baseline and baseline.correct else None
        b_memory = baseline.memory if baseline and baseline.correct else None
        b_cost = baseline.cost if baseline and baseline.correct else None
        b_score = baseline.score if baseline and baseline.correct else None

        a_params = chosen.params if chosen else None
        a_memory = chosen.memory if chosen else None
        a_cost = chosen.cost if chosen else None
        a_score = chosen.score if chosen else None

        # When proposal-002 passes changed the winner, the realized output is
        # the post-002 model: surface its cost/score as the "after" values and
        # attribute the gain to the applied G-passes.
        if res.p002_applied and res.cost_post_002 is not None:
            a_cost = res.cost_post_002
            a_score = res.score_post_002
            if res.score_pre_002 is not None and res.score_post_002 is not None:
                gain = res.score_post_002 - res.score_pre_002
                for name in res.p002_applied:
                    p002_gain[name] = p002_gain.get(name, 0.0) + gain / len(
                        res.p002_applied
                    )

        if b_score is not None:
            total_score_before += b_score
        if a_score is not None:
            total_score_after += a_score

        applied_parts = list(chosen.passes_applied) if chosen else []
        applied_parts += res.p002_applied
        reverted_parts = list(chosen.passes_reverted) if chosen else []
        reverted_parts += res.p002_reverted
        applied = ",".join(applied_parts) if applied_parts else "-"
        reverted = ",".join(reverted_parts) if reverted_parts else "-"

        rows.append(
            "| {task:03d} | {src} | {bp} / {ap} | {bm} / {am} | "
            "{bc} / {ac} | {bs} / {as_} | {ap_} | {rv} |".format(
                task=res.task_num,
                src=res.chosen_source or "FALLBACK-A",
                bp=_fmt(b_params),
                ap=_fmt(a_params),
                bm=_fmt(b_memory),
                am=_fmt(a_memory),
                bc=_fmt(b_cost),
                ac=_fmt(a_cost),
                bs=_fmt(b_score),
                as_=_fmt(a_score),
                ap_=applied,
                rv=reverted or "-",
            )
        )

        if res.warning:
            warnings.append(f"Task {res.task_num:03d}: {res.warning}")

        json_tasks.append(
            {
                "task": res.task_num,
                "chosen_source": res.chosen_source,
                "fallback_used": res.fallback_used,
                "divergent_local": res.divergent_local,
                "warning": res.warning,
                "baseline": _candidate_to_json(baseline),
                "chosen": _candidate_to_json(chosen),
                "p002_applied": res.p002_applied,
                "p002_reverted": res.p002_reverted,
                "cost_pre_002": res.cost_pre_002,
                "cost_post_002": res.cost_post_002,
                "score_pre_002": res.score_pre_002,
                "score_post_002": res.score_post_002,
                "margin_used": res.margin_used,
                "candidates": [
                    {
                        **_candidate_to_json(c),
                        "passes_applied": c.passes_applied,
                        "passes_reverted": c.passes_reverted,
                        "note": c.note,
                    }
                    for c in res.candidates
                ],
            }
        )

    s3_reverts = [
        f"Task {res.task_num:03d} (source {c.source})"
        for res in results
        for c in res.candidates
        if "S3" in c.passes_reverted
    ]

    # Proposal-002 pass accounting.
    p002_applied_tasks: dict[str, list[int]] = {"G3": [], "G2": [], "G1": []}
    p002_reverted_tasks: list[str] = []
    for res in results:
        for name in res.p002_applied:
            p002_applied_tasks.setdefault(name, []).append(res.task_num)
        for name in res.p002_reverted:
            p002_reverted_tasks.append(
                f"Task {res.task_num:03d}: {name} reverted "
                f"(gate failed: margin/cost/identity)"
            )

    divergent = [res for res in results if res.divergent_local]
    divergent_entries: list[str] = []
    for res in divergent:
        chosen = res.chosen
        if res.chosen_source == "A(identity)" and chosen is not None:
            detail = (
                f"identity-verified optimized A "
                f"(cost {chosen.cost}, score {chosen.score:.4f}); "
                "baseline comparison unavailable"
            )
        else:
            detail = (
                "kept original A unchanged; "
                "baseline + cost comparison unavailable"
            )
        divergent_entries.append(
            f"Task {res.task_num:03d}: {res.chosen_source} — {detail}"
        )

    md_lines = _build_markdown(
        run_number,
        sources,
        passes,
        cfg,
        rows,
        total_score_before,
        total_score_after,
        s3_reverts,
        p002_applied_tasks,
        p002_reverted_tasks,
        p002_gain,
        divergent_entries,
        warnings,
    )

    json_payload: dict[str, Any] = {
        "run": run_number,
        "passes": passes,
        "sources": sources,
        "skip_tasks_s3": cfg.skip_tasks_s3,
        "g1_fp16_tasks": cfg.g1_fp16_tasks,
        "g1_g2_margin": cfg.g2_margin,
        "totals": {
            "score_before": total_score_before,
            "score_after": total_score_after,
            "score_delta": total_score_after - total_score_before,
        },
        "p002_summary": {
            "applied_tasks": {
                k: sorted(v) for k, v in p002_applied_tasks.items()
            },
            "realized_gain": p002_gain,
            "reverts": p002_reverted_tasks,
        },
        "s3_reverts": s3_reverts,
        "divergent_local_tasks": [
            {
                "task": res.task_num,
                "chosen_source": res.chosen_source,
                "resolution": (
                    "identity"
                    if res.chosen_source == "A(identity)"
                    else "orig-divergent"
                ),
                "cost": res.chosen.cost if res.chosen else None,
                "score": res.chosen.score if res.chosen else None,
            }
            for res in divergent
        ],
        "warnings": warnings,
        "tasks": json_tasks,
    }
    return "\n".join(md_lines), json_payload


def _build_markdown(
    run_number: int,
    sources: list[str],
    passes: str,
    cfg: ReportConfig,
    rows: list[str],
    total_score_before: float,
    total_score_after: float,
    s3_reverts: list[str],
    p002_applied_tasks: dict[str, list[int]],
    p002_reverted_tasks: list[str],
    p002_gain: dict[str, float],
    divergent_entries: list[str],
    warnings: list[str],
) -> list[str]:
    md: list[str] = []
    md.append(f"# NeuroGolf optimization report — run {run_number:03d}")
    md.append("")
    md.append(f"- Sources considered: {', '.join(sources)}")
    md.append(f"- Pass set: {passes}")
    md.append(
        f"- Passes: S1, S2, S4 always; S3 except tasks {cfg.skip_tasks_s3}"
    )
    if passes == cfg.passes_002:
        md.append(
            f"- Residual 002 passes: G3 (no-op) then G2 (FLOAT->BOOL) then "
            f"G1 (FP16, tasks {cfg.g1_fp16_tasks}); G1/G2 margin = {cfg.g2_margin}"
        )
    md.append("")
    md.append("## Totals (expected score = max(1, 25 - ln(cost)))")
    md.append("")
    md.append(f"- Sum of scores BEFORE (baseline A): {total_score_before:.4f}")
    md.append(f"- Sum of scores AFTER (optimized):  {total_score_after:.4f}")
    md.append(f"- Delta: {total_score_after - total_score_before:+.4f}")
    md.append("")
    md.append("## Per-task results")
    md.append("")
    md.append(
        "| Task | Source | Params (before / after) | Memory (before / after) "
        "| Cost (before / after) | Score (before / after) | Passes applied | Reverted |"
    )
    md.append("|---|---|---|---|---|---|---|---|")
    md.extend(rows)
    md.append("")
    md.append("## S3 reverts")
    md.append("")
    md.extend(f"- {e}" for e in s3_reverts) if s3_reverts else md.append("- None")
    md.append("")
    if passes == cfg.passes_002:
        md.append("## Proposal 002 residual passes")
        md.append("")
        md.append(f"- G1/G2 margin used: {cfg.g2_margin}")
        for name in ("G3", "G2", "G1"):
            tasks_n = p002_applied_tasks.get(name, [])
            md.append(
                f"- {name}: applied on {len(tasks_n)} tasks "
                f"(realized gain {p002_gain.get(name, 0.0):+.4f}); "
                f"tasks {sorted(tasks_n) if tasks_n else 'none'}"
            )
        md.append("")
        md.append("### 002 reverts (gate failures)")
        md.append("")
        if p002_reverted_tasks:
            md.extend(f"- {e}" for e in p002_reverted_tasks)
        else:
            md.append("- None")
        md.append("")
    md.append("## Locally-divergent tasks")
    md.append("")
    md.append(
        "Tasks where baseline source A fails LOCAL validation but is known to "
        "pass the official grader (float `> 0.0` boundary divergence). No "
        "source swap is performed; source A is kept (optimized only when "
        "bit-identical to the original)."
    )
    md.append("")
    if divergent_entries:
        md.extend(f"- {e}" for e in divergent_entries)
    else:
        md.append("- None")
    md.append("")
    md.append("## Failures / warnings")
    md.append("")
    if warnings:
        md.extend(f"- {e}" for e in warnings)
    else:
        md.append("- None")
    md.append("")
    return md


def write_reports(
    md_text: str,
    json_payload: dict[str, Any],
    run_number: int,
    reports_dir: Path,
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"run-{run_number:03d}.md"
    json_path = reports_dir / f"run-{run_number:03d}.json"
    md_path.write_text(md_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return md_path, json_path


def build_submission_zip(
    task_nums: list[int], optimized_dir: Path, submission_zip: Path
) -> Path:
    """Zip the optimized task files flat at the archive root (deterministic)."""
    submission_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for task_num in sorted(task_nums):
            path = optimized_dir / f"task{task_num:03d}.onnx"
            if path.is_file():
                zf.write(path, arcname=f"task{task_num:03d}.onnx")
    return submission_zip
