"""Per-task merge of new external submissions into the current best set.

For every task 1..400 this compares the grader-proven INCUMBENT
(``artifacts/optimized/taskNNN.onnx``, whose current cost/score are rescored
from the actual file; ``artifacts/reports/run-012.json`` is only a fallback
when rescoring fails) against external CHALLENGER sources. By default, these
are the original submission13 sources:

* ``E`` = ``inputs/submission13/submission/overrides/taskNNN.onnx``
* ``F`` = ``inputs/submission13/submission/base_submission/taskNNN.onnx``

Use repeated ``--source NAME=PATH`` arguments to replace the defaults with
arbitrary read-only challenger directories.

Each challenger is run through the SAME proposal-001/002 pass stack as a normal
``--passes 002`` source — S1, S2, S4, then S3 (except ``SKIP_TASKS``) via
``optimize_submission._apply_pipeline``, then the residual G3/G2/G1 passes via
``optimize_submission._apply_002_passes`` (``allow_dtype=True``) — and is only
allowed to replace the incumbent when ALL of these hold:

  a. it is locally gold-correct (``score_and_verify`` with ``require_correct=True``),
  b. it is margin-stable: no raw output cell lies in the open interval
     ``(0, 0.25)`` on any example (``scoring.model_margin_stable``),
  c. its cost is strictly less than the incumbent cost.

The winner is the cheapest accepted challenger, else the incumbent (which is
always eligible — it is grader-proven and never needs local validation). The
winner is written to ``artifacts/optimized/taskNNN.onnx``; when the incumbent
wins its file is left untouched (byte-identical).

SAFETY: ``inputs/`` is never modified. Back up the pre-merge
``artifacts/optimized`` to ``artifacts/optimized_pre_merge/`` BEFORE running
this (the caller does ``cp -r``). Deterministic: no randomness in selection, no
timestamps in outputs. No new optimization ideas are introduced here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))

import optimize_submission as pipe  # noqa: E402  (path insert above)
from lib import scoring  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
OPTIMIZED_DIR = ARTIFACTS / "optimized"
PRE_MERGE_DIR = ARTIFACTS / "optimized_pre_merge"
REPORTS_DIR = ARTIFACTS / "reports"
SUBMISSION_ZIP = ARTIFACTS / "submission.zip"
INCUMBENT_REPORT = REPORTS_DIR / "run-012.json"

# Default external challenger sources (READ-ONLY).
DEFAULT_CHALLENGER_DIRS: dict[str, Path] = {
    "E": REPO_ROOT / "inputs" / "submission13" / "submission" / "overrides",
    "F": REPO_ROOT / "inputs" / "submission13" / "submission" / "base_submission",
}
CHALLENGER_DIRS: dict[str, Path] = dict(DEFAULT_CHALLENGER_DIRS)
CHALLENGER_ORDER = list(CHALLENGER_DIRS)

NUM_TASKS = pipe.NUM_TASKS
MARGIN = pipe.G2_MARGIN  # 0.25, the open-interval boundary-stability margin.

MERGE_MD = REPORTS_DIR / "merge-001.md"
MERGE_JSON = REPORTS_DIR / "merge-001.json"


# --- Data carriers ------------------------------------------------------------


@dataclass
class ChallengerEval:
    """Outcome of evaluating one challenger source for one task."""

    source: str
    present: bool
    accepted: bool
    # Rejection reason when not accepted: 'missing' / 'incorrect' / 'margin'
    # / 'not-cheaper' / 'unscorable'. Empty when accepted.
    reject_reason: str = ""
    cost: int | None = None
    score: float | None = None
    margin_min: float | None = None
    passes_applied: list[str] = field(default_factory=list)
    p002_applied: list[str] = field(default_factory=list)
    model: onnx.ModelProto | None = None


@dataclass
class MergeResult:
    """Aggregated merge decision for one task."""

    task_num: int
    winner: str  # 'incumbent' or one of CHALLENGER_ORDER
    incumbent_cost: int
    incumbent_score: float
    incumbent_divergent: bool
    cost_before: int
    cost_after: int
    score_before: float
    score_after: float
    margin_min: float | None  # observed margin minimum of the winner (None = incumbent kept)
    challengers: list[ChallengerEval]
    changed: bool


# --- Source configuration -----------------------------------------------------


def _source_name_valid(name: str) -> bool:
    return bool(name) and all("A" <= c <= "Z" for c in name)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_source_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def configure_challenger_sources(
    source_specs: list[str] | None, parser: argparse.ArgumentParser
) -> None:
    """Configure global challenger sources from CLI specs or defaults."""
    global CHALLENGER_DIRS, CHALLENGER_ORDER

    if source_specs:
        sources: dict[str, Path] = {}
        for spec in source_specs:
            if "=" not in spec:
                parser.error(
                    f"--source must be NAME=PATH with NAME uppercase letters: {spec}"
                )
            name, raw_path = spec.split("=", 1)
            if not _source_name_valid(name):
                parser.error(
                    f"--source NAME must use uppercase ASCII letters only: {name}"
                )
            if not raw_path:
                parser.error(f"--source {name}=PATH is missing PATH")
            if name in sources:
                parser.error(f"--source NAME is duplicated: {name}")
            sources[name] = _resolve_source_path(raw_path)
    else:
        sources = dict(DEFAULT_CHALLENGER_DIRS)

    for name, path in sources.items():
        if not path.is_dir():
            parser.error(
                f"--source {name} path is not a directory: {_display_path(path)}"
            )
        if not any(path.glob("task*.onnx")):
            parser.error(
                f"--source {name} path contains no task*.onnx files: "
                f"{_display_path(path)}"
            )

    CHALLENGER_DIRS = sources
    CHALLENGER_ORDER = list(sources)


# --- Incumbent loading --------------------------------------------------------


def load_incumbents() -> dict[int, dict]:
    """Load fallback per-task incumbent metadata from run-012.json.

    This report is not authoritative for current costs because
    ``artifacts/optimized`` may have been improved since it was written.
    ``merge_task`` rescoring the actual incumbent file is the primary path; this
    data is used only if that rescore returns ``None``. Returns
    ``{task_num: {"cost", "score", "divergent"}}``.
    """
    with open(INCUMBENT_REPORT) as f:
        rep = json.load(f)
    out: dict[int, dict] = {}
    for t in rep["tasks"]:
        out[t["task"]] = {
            "cost": t["cost_post_002"],
            "score": t["score_post_002"],
            "divergent": bool(t.get("divergent_local", False)),
        }
    return out


def _rescore_incumbent_file(task_num: int, workdir: str) -> tuple[int, float] | None:
    """Current cost/score for the written incumbent file.

    Scores the written incumbent file with ``require_correct=False`` so a
    locally-divergent incumbent still yields its cost (it is grader-proven).
    """
    path = OPTIMIZED_DIR / f"task{task_num:03d}.onnx"
    if not path.is_file():
        return None
    try:
        model = onnx.load(str(path))
    except Exception:
        return None
    scored = scoring.score_and_verify(
        model, task_num, workdir, label="incumbent", require_correct=False
    )
    if scored is None:
        return None
    return scored["cost"], scored["score"]


def _fallback_incumbent_from_report(
    task_num: int, incumbents: dict[int, dict]
) -> tuple[int, float, bool]:
    """Use stale run-012 metadata only when rescoring the incumbent fails."""
    inc = incumbents.get(task_num)
    if inc is None:
        raise RuntimeError(
            f"task{task_num:03d}: incumbent rescore failed and no "
            f"{INCUMBENT_REPORT.name} fallback exists"
        )
    print(
        f"WARNING: task{task_num:03d}: incumbent rescore returned None; "
        f"falling back to {INCUMBENT_REPORT.name} cost {inc['cost']}.",
        file=sys.stderr,
    )
    return inc["cost"], inc["score"], inc["divergent"]


# --- Challenger evaluation ----------------------------------------------------


def evaluate_challenger(
    source: str, task_num: int, incumbent_cost: int, workdir: str
) -> ChallengerEval:
    """Apply the full 002 pass stack to a challenger and run the acceptance gate.

    Reuses ``optimize_submission._apply_pipeline`` (S1,S2,S4,S3 with
    ``SKIP_TASKS``) then ``_apply_002_passes`` (G3 bit-identity gate, then G2/G1
    mask+margin gates, ``allow_dtype=True``) exactly as ``--passes 002`` does for
    a normal source. Then the challenger acceptance gate:

    a. locally gold-correct via ``score_and_verify(require_correct=True)``,
    b. margin-stable via ``scoring.model_margin_stable`` (no raw cell in
       ``(0, 0.25)``),
    c. cost strictly < ``incumbent_cost``.

    The margin minimum is always recorded (even on rejection) for the report.
    """
    src_path = CHALLENGER_DIRS[source] / f"task{task_num:03d}.onnx"
    if not src_path.is_file():
        return ChallengerEval(source=source, present=False, accepted=False,
                              reject_reason="missing")

    base = onnx.load(str(src_path))
    # S-stack (S1,S2,S4,S3) with per-pass revert — identical to a normal source.
    s_model, s_applied, _s_reverted = pipe._apply_pipeline(base, task_num)
    # Residual 002 passes (G3 -> G2 -> G1) with the proposal-002 per-task gates.
    outcome = pipe._apply_002_passes(s_model, task_num, workdir, allow_dtype=True)
    model = outcome.model
    p002_applied = list(outcome.applied)

    # (a) Local gold correctness AND scorability/cost in one call.
    scored = scoring.score_and_verify(
        model, task_num, workdir, label=f"{source}chal", require_correct=True
    )

    # Margin minimum is independent of correctness; record it for the report.
    stable, margin_min = scoring.model_margin_stable(model, task_num, MARGIN)

    if scored is None:
        return ChallengerEval(
            source=source, present=True, accepted=False,
            reject_reason="incorrect", margin_min=margin_min,
            passes_applied=s_applied, p002_applied=p002_applied,
        )

    cost = scored["cost"]
    score = scored["score"]

    # (b) Margin stability.
    if not stable:
        return ChallengerEval(
            source=source, present=True, accepted=False,
            reject_reason="margin", cost=cost, score=score,
            margin_min=margin_min, passes_applied=s_applied,
            p002_applied=p002_applied,
        )

    # (c) Strictly cheaper than the incumbent.
    if cost >= incumbent_cost:
        return ChallengerEval(
            source=source, present=True, accepted=False,
            reject_reason="not-cheaper", cost=cost, score=score,
            margin_min=margin_min, passes_applied=s_applied,
            p002_applied=p002_applied,
        )

    return ChallengerEval(
        source=source, present=True, accepted=True, reject_reason="",
        cost=cost, score=score, margin_min=margin_min,
        passes_applied=s_applied, p002_applied=p002_applied, model=model,
    )


# --- Per-task merge -----------------------------------------------------------


def merge_task(
    task_num: int, incumbents: dict[int, dict], workdir: str
) -> MergeResult:
    """Decide the winner for one task and write it to ``artifacts/optimized/``."""
    inc = incumbents.get(task_num)
    inc_divergent = bool(inc.get("divergent", False)) if inc is not None else False

    rescored = _rescore_incumbent_file(task_num, workdir)
    if rescored is not None:
        inc_cost, inc_score = rescored
    else:
        inc_cost, inc_score, inc_divergent = _fallback_incumbent_from_report(
            task_num, incumbents
        )

    evals: list[ChallengerEval] = [
        evaluate_challenger(s, task_num, inc_cost, workdir)
        for s in CHALLENGER_ORDER
    ]

    accepted = [e for e in evals if e.accepted and e.cost is not None]
    # Winner = cheapest accepted challenger; tie-break by CLI/default source order.
    if accepted:
        winner_eval = min(
            accepted, key=lambda e: (e.cost, CHALLENGER_ORDER.index(e.source))
        )
        # Strictly cheaper is guaranteed by the gate; defensive re-check.
        if winner_eval.cost is not None and winner_eval.cost < inc_cost:
            _save_winner(task_num, winner_eval.model)
            return MergeResult(
                task_num=task_num, winner=winner_eval.source,
                incumbent_cost=inc_cost, incumbent_score=inc_score,
                incumbent_divergent=inc_divergent,
                cost_before=inc_cost, cost_after=winner_eval.cost,
                score_before=inc_score, score_after=winner_eval.score,
                margin_min=winner_eval.margin_min, challengers=evals,
                changed=True,
            )

    # Incumbent kept: leave the existing file untouched (byte-identical).
    return MergeResult(
        task_num=task_num, winner="incumbent",
        incumbent_cost=inc_cost, incumbent_score=inc_score,
        incumbent_divergent=inc_divergent,
        cost_before=inc_cost, cost_after=inc_cost,
        score_before=inc_score, score_after=inc_score,
        margin_min=None, challengers=evals, changed=False,
    )


def _save_winner(task_num: int, model: onnx.ModelProto | None) -> None:
    if model is None:
        raise RuntimeError(f"task{task_num:03d}: winner model is None")
    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OPTIMIZED_DIR / f"task{task_num:03d}.onnx"))


# --- Reporting ----------------------------------------------------------------


def _challenger_json(e: ChallengerEval) -> dict:
    return {
        "source": e.source,
        "present": e.present,
        "accepted": e.accepted,
        "reject_reason": e.reject_reason,
        "cost": e.cost,
        "score": e.score,
        "margin_min": e.margin_min,
        "passes_applied": e.passes_applied,
        "p002_applied": e.p002_applied,
    }


def build_report(results: list[MergeResult]) -> tuple[str, dict]:
    """Build the merge markdown + JSON report.

    Totals: expected score before = sum of rescored incumbent scores; after =
    sum over all tasks of the realized winner score (incumbent score for kept
    tasks, challenger scorer result for adopted tasks).
    """
    total_before = sum(r.score_before for r in results)
    total_after = sum(r.score_after for r in results)

    adopt_counts = {source: 0 for source in CHALLENGER_ORDER}
    adopt_counts["incumbent"] = 0
    for r in results:
        adopt_counts[r.winner] += 1

    # Rejection histogram over every challenger evaluation.
    reject_hist: dict[str, int] = {
        "incorrect": 0, "margin": 0, "not-cheaper": 0,
        "missing": 0, "unscorable": 0,
    }
    for r in results:
        for e in r.challengers:
            if not e.accepted and e.reject_reason:
                reject_hist[e.reject_reason] = (
                    reject_hist.get(e.reject_reason, 0) + 1
                )

    # Tasks where an external candidate was strictly cheaper but rejected by the
    # margin gate (candidates for manual review).
    margin_rejects: list[dict] = []
    for r in results:
        for e in r.challengers:
            if (
                e.reject_reason == "margin"
                and e.cost is not None
                and e.cost < r.incumbent_cost
            ):
                margin_rejects.append({
                    "task": r.task_num,
                    "source": e.source,
                    "challenger_cost": e.cost,
                    "incumbent_cost": r.incumbent_cost,
                    "margin_min": e.margin_min,
                })

    changed = [r for r in results if r.changed]
    # Top gains by realized score delta (adopted tasks only).
    gains = sorted(
        ({
            "task": r.task_num, "winner": r.winner,
            "cost_before": r.cost_before, "cost_after": r.cost_after,
            "score_before": r.score_before, "score_after": r.score_after,
            "score_delta": r.score_after - r.score_before,
        } for r in changed),
        key=lambda d: d["score_delta"], reverse=True,
    )

    json_payload = {
        "incumbent_report": INCUMBENT_REPORT.name,
        "challenger_sources": {
            source: _display_path(path)
            for source, path in CHALLENGER_DIRS.items()
        },
        "margin": MARGIN,
        "totals": {
            "score_before": total_before,
            "score_after": total_after,
            "score_delta": total_after - total_before,
            "cost_before": sum(r.cost_before for r in results),
            "cost_after": sum(r.cost_after for r in results),
        },
        "adoption_counts": adopt_counts,
        "rejection_histogram": reject_hist,
        "margin_rejected_cheaper": margin_rejects,
        "changed_tasks": [
            {
                "task": r.task_num,
                "winner": r.winner,
                "incumbent_cost": r.incumbent_cost,
                "incumbent_divergent": r.incumbent_divergent,
                "cost_before": r.cost_before,
                "cost_after": r.cost_after,
                "score_before": r.score_before,
                "score_after": r.score_after,
                "score_delta": r.score_after - r.score_before,
                "margin_min": r.margin_min,
                "challengers": [_challenger_json(e) for e in r.challengers],
            }
            for r in changed
        ],
        "all_tasks": [
            {
                "task": r.task_num,
                "winner": r.winner,
                "incumbent_divergent": r.incumbent_divergent,
                "cost_before": r.cost_before,
                "cost_after": r.cost_after,
                "score_before": r.score_before,
                "score_after": r.score_after,
                "challengers": [_challenger_json(e) for e in r.challengers],
            }
            for r in results
        ],
    }

    md = _build_merge_markdown(
        results, total_before, total_after, adopt_counts, reject_hist,
        margin_rejects, changed, gains,
    )
    return "\n".join(md), json_payload


def _build_merge_markdown(
    results: list[MergeResult],
    total_before: float,
    total_after: float,
    adopt_counts: dict[str, int],
    reject_hist: dict[str, int],
    margin_rejects: list[dict],
    changed: list[MergeResult],
    gains: list[dict],
) -> list[str]:
    md: list[str] = []
    md.append(f"# NeuroGolf external merge report — {MERGE_MD.stem}")
    md.append("")
    source_desc = ", ".join(
        f"{source} = `{_display_path(path)}`"
        for source, path in CHALLENGER_DIRS.items()
    )
    md.append(
        "Per-task merge of external challenger sources "
        f"({source_desc}) into the grader-proven incumbent set "
        "(`artifacts/optimized/`, costs rescored from current files; "
        "run-012.json fallback only if rescoring fails)."
    )
    md.append("")
    md.append(
        "Acceptance gate for a challenger (ALL required): "
        "(a) locally gold-correct, (b) margin-stable — no raw cell in "
        f"(0, {MARGIN}) — (c) cost strictly < incumbent. Incumbent is always "
        "eligible (grader-proven) and is rescored without requiring local "
        "gold-correctness."
    )
    md.append("")
    md.append("## Totals (expected score = max(1, 25 - ln(cost)))")
    md.append("")
    md.append(f"- Tasks: {len(results)}")
    md.append(f"- Sum of scores BEFORE (incumbent): {total_before:.4f}")
    md.append(f"- Sum of scores AFTER  (merged):    {total_after:.4f}")
    md.append(f"- Delta: {total_after - total_before:+.4f}")
    md.append("")
    md.append("## Adoption counts")
    md.append("")
    for source, path in CHALLENGER_DIRS.items():
        md.append(
            f"- {source} ({_display_path(path)}): {adopt_counts.get(source, 0)}"
        )
    md.append(f"- incumbent kept: {adopt_counts['incumbent']}")
    md.append("")
    md.append("## Rejection histogram (over all challenger evaluations)")
    md.append("")
    for reason in ("incorrect", "margin", "not-cheaper", "missing", "unscorable"):
        md.append(f"- {reason}: {reject_hist.get(reason, 0)}")
    md.append("")
    md.append("## Cheaper-but-margin-rejected (manual-review candidates)")
    md.append("")
    md.append(
        "External candidate was strictly cheaper than the incumbent but failed "
        "the margin gate (a raw cell in (0, "
        f"{MARGIN})). NOT adopted — these may pass the official grader but are "
        "platform-flip risks locally."
    )
    md.append("")
    if margin_rejects:
        md.append("| Task | Src | Challenger cost | Incumbent cost | Margin min |")
        md.append("|---|---|---|---|---|")
        for e in sorted(margin_rejects, key=lambda d: d["task"]):
            mm = "-" if e["margin_min"] is None else f"{e['margin_min']:.4f}"
            md.append(
                f"| {e['task']:03d} | {e['source']} | {e['challenger_cost']} "
                f"| {e['incumbent_cost']} | {mm} |"
            )
    else:
        md.append("- None")
    md.append("")
    md.append("## Changed tasks (adopted external)")
    md.append("")
    if changed:
        md.append(
            "| Task | Winner | Cost before | Cost after | Score before "
            "| Score after | Score delta | Margin min |"
        )
        md.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(changed, key=lambda x: x.task_num):
            mm = "-" if r.margin_min is None else f"{r.margin_min:.4f}"
            md.append(
                f"| {r.task_num:03d} | {r.winner} | {r.cost_before} "
                f"| {r.cost_after} | {r.score_before:.4f} | {r.score_after:.4f} "
                f"| {r.score_after - r.score_before:+.4f} | {mm} |"
            )
    else:
        md.append("- None")
    md.append("")
    md.append("## Top-10 task gains")
    md.append("")
    if gains:
        md.append("| Task | Winner | Cost before -> after | Score delta |")
        md.append("|---|---|---|---|")
        for g in gains[:10]:
            md.append(
                f"| {g['task']:03d} | {g['winner']} | "
                f"{g['cost_before']} -> {g['cost_after']} | "
                f"{g['score_delta']:+.4f} |"
            )
    else:
        md.append("- None")
    md.append("")
    return md


def write_report(md_text: str, json_payload: dict) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MERGE_MD.write_text(md_text, encoding="utf-8")
    MERGE_JSON.write_text(
        json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return MERGE_MD, MERGE_JSON


def build_submission_zip(task_nums: list[int]) -> Path:
    """Zip all optimized task files flat at the archive root (deterministic)."""
    SUBMISSION_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for task_num in sorted(task_nums):
            path = OPTIMIZED_DIR / f"task{task_num:03d}.onnx"
            if path.is_file():
                zf.write(path, arcname=f"task{task_num:03d}.onnx")
    return SUBMISSION_ZIP


# --- Entry point --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-task merge of external submissions into the best set."
    )
    parser.add_argument(
        "--tasks", default="all",
        help="Task selection: 'all', a range like '1-5', or a list '7,12,300'.",
    )
    parser.add_argument(
        "--zip", action="store_true",
        help="Rebuild artifacts/submission.zip (flat task*.onnx at root).",
    )
    parser.add_argument(
        "--source", action="append", metavar="NAME=PATH",
        help=(
            "Challenger source directory. Repeat to merge multiple sources. "
            "NAME must be uppercase ASCII letters; PATH may be relative to "
            "the repo root or absolute. Overrides the default E/F sources."
        ),
    )
    args = parser.parse_args(argv)
    configure_challenger_sources(args.source, parser)

    if not PRE_MERGE_DIR.is_dir():
        print(
            "ERROR: artifacts/optimized_pre_merge/ is missing. Back up the "
            "pre-merge optimized dir (cp -r) before running the merge."
        )
        return 1

    task_nums = pipe.parse_tasks(args.tasks)
    if not task_nums:
        print("No tasks selected.")
        return 1

    incumbents = load_incumbents()
    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)
    results: list[MergeResult] = []

    with tempfile.TemporaryDirectory(prefix="neurogolf_merge_") as workdir:
        total = len(task_nums)
        for idx, task_num in enumerate(task_nums, start=1):
            r = merge_task(task_num, incumbents, workdir)
            results.append(r)
            tag = "ADOPT " + r.winner if r.changed else "keep  incumbent"
            print(
                f"[{idx}/{total}] task{task_num:03d} {tag} "
                f"cost {r.cost_before} -> {r.cost_after} "
                f"score {r.score_before:.3f} -> {r.score_after:.3f}"
            )

    md_text, json_payload = build_report(results)
    md_path, json_path = write_report(md_text, json_payload)
    print(f"Report: {md_path}")
    print(f"JSON:   {json_path}")
    print(
        f"Totals: before {json_payload['totals']['score_before']:.4f} -> "
        f"after {json_payload['totals']['score_after']:.4f} "
        f"(delta {json_payload['totals']['score_delta']:+.4f})"
    )
    print(f"Adoption: {json_payload['adoption_counts']}")

    if args.zip:
        zip_path = build_submission_zip(list(range(1, NUM_TASKS + 1)))
        print(f"Submission zip: {zip_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
