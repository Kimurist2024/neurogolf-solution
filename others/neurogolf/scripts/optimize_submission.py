"""NeuroGolf 2026 — behaviour-preserving ONNX cost-reduction pipeline.

Implements the pipeline from ``proposals/001-zero-risk-onnx-cost-reduction.md``:
for every task, load each of the three candidate sources, apply passes
S1/S2/S4 (always) and S3 (except for the known grader-incompatible tasks),
validate and score with the ported official scorer, and pick the lowest-cost
correct candidate. Results are written to ``artifacts/optimized/`` along with a
markdown + JSON report; ``--zip`` additionally builds a flat submission zip.

Deterministic: no randomness in selection, no timestamps in outputs. The run
number is the next free integer for the report filenames.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import dtype_passes as dtype_opt
from lib import optimizations as opt
from lib import reporting
from lib import scoring
from lib.pipeline_types import CandidateResult, P002Outcome, TaskResult

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
OPTIMIZED_DIR = ARTIFACTS / "optimized"
REPORTS_DIR = ARTIFACTS / "reports"
SUBMISSION_ZIP = ARTIFACTS / "submission.zip"

# Candidate sources A, B, C (proposal section 2).
SOURCE_DIRS: dict[str, Path] = {
    "A": REPO_ROOT / "inputs" / "neurogolf-6347-80" / "overrides",
    "B": REPO_ROOT / "inputs" / "neurogolf-6347-80" / "base_submission",
    "C": REPO_ROOT / "inputs" / "neurogolf-6347-76",
}
DEFAULT_SOURCES = ["A", "B", "C"]
BASELINE_SOURCE = "A"

# Tasks where S3 (scalar compression) passes locally but fails the official
# grader; the proposal mandates skipping S3 for them.
SKIP_TASKS: frozenset[int] = frozenset({54, 158})

NUM_TASKS = 400

# --- Proposal 002 constants ---------------------------------------------------

# Selective FP16 (G1) applies ONLY to the three census tasks where the FP16
# delta is strictly positive (docs/research/s6-memory-census.md, top of table).
G1_FP16_TASKS: frozenset[int] = frozenset({170, 97, 64})

# G1/G2 mask-equality margin: the converted model must keep every "on" output
# cell at least this far from the > 0.0 boundary (proposal section 3).
G2_MARGIN: float = 0.25

# Valid --passes selections.
PASSES_001 = "001"
PASSES_002 = "002"
DEFAULT_PASSES = PASSES_002


def _validate_pass(model: onnx.ModelProto) -> None:
    """Raise if the model fails the checker or strict shape inference."""
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)


def _apply_pipeline(
    base_model: onnx.ModelProto, task_num: int
) -> tuple[onnx.ModelProto, list[str], list[str]]:
    """Apply S1, S2, S4 (always) and S3 (unless skipped) with per-pass revert.

    Each pass is applied to the running model, then validated with
    ``onnx.checker.check_model`` + ``infer_shapes(strict_mode=True)``. If a pass
    raises, only that pass is reverted and the pipeline continues from the
    pre-pass model.
    """
    passes: list[tuple[str, Callable[[onnx.ModelProto], tuple[onnx.ModelProto, Any]]]]
    passes = [
        ("S1", opt.s1_prune_unused_initializers),
        ("S2", opt.s2_dedup_initializers),
        ("S4", opt.s4_clean_value_info),
    ]
    if task_num not in SKIP_TASKS:
        passes.append(("S3", opt.s3_compress_uniform_to_scalar))

    current = base_model
    applied: list[str] = []
    reverted: list[str] = []
    for name, func in passes:
        try:
            candidate, _stats = func(current)
            _validate_pass(candidate)
            current = candidate
            applied.append(name)
        except Exception:  # noqa: BLE001 — revert this pass only, keep going.
            reverted.append(name)
    return current, applied, reverted


def _cost_of(model: onnx.ModelProto, task_num: int, workdir: str, label: str) -> tuple[int | None, float | None]:
    """Return ``(cost, score)`` for a model, ignoring local gold correctness.

    Uses ``require_correct=False`` so a locally-divergent gold mismatch does not
    suppress the cost (safety is enforced separately by the G3 bit-identity or
    the G1/G2 mask+margin gates). Returns ``(None, None)`` on any hard rejection
    (file size, load failure, unscorable, negative).
    """
    scored = scoring.score_and_verify(
        model, task_num, workdir, label=label, require_correct=False
    )
    if scored is None:
        return None, None
    return scored["cost"], scored["score"]


def _apply_002_passes(
    winner: onnx.ModelProto,
    task_num: int,
    workdir: str,
    *,
    allow_dtype: bool,
) -> P002Outcome:
    """Apply G3 -> G2 -> G1 to a winner model with per-task verification gates.

    Order (proposal pipeline integration): G3 (no-op removal) first, then G2
    (FLOAT->BOOL), then G1 (selective FP16, only on the three G1 tasks).

    Gates (proposal section 3), each applied per pass; a failing pass is
    reverted to its pre-pass model and the pipeline continues:

    * **G3**: must be bit-identical to the pre-G3 model
      (``outputs_bit_identical``) AND not increase cost. Allowed even on
      divergent-local tasks (identity is platform-proof).
    * **G2 / G1**: numeric output changes, so require, on every example,
      ``masks_equal_with_margin`` (thresholded masks identical + no output cell
      in the open ``(0, margin)`` interval) AND a strict cost decrease. Skipped
      entirely when ``allow_dtype`` is ``False`` (divergent-local tasks).

    The returned ``model`` is the SAME object as ``winner`` when nothing was
    applied (so untouched tasks stay byte-identical to their 001 winner).
    """
    cost_before, score_before = _cost_of(winner, task_num, workdir, "p002base")

    current = winner
    applied: list[str] = []
    reverted: list[str] = []

    # --- G3: no-op removal (bit-identity gate) ---
    try:
        g3_model, g3_stats = opt.g3_remove_noops(current)
        _validate_pass(g3_model)
        bit_ok = scoring.outputs_bit_identical(current, g3_model, task_num)
        cost_g3, _ = _cost_of(g3_model, task_num, workdir, "p002g3")
        cur_cost, _ = _cost_of(current, task_num, workdir, "p002g3pre")
        cost_ok = (
            cost_g3 is not None
            and cur_cost is not None
            and cost_g3 < cur_cost
        )
        if bit_ok and cost_ok and g3_stats["removed_nodes"] > 0:
            current = g3_model
            applied.append("G3")
        elif g3_stats["removed_nodes"] > 0:
            reverted.append("G3")
    except Exception:  # noqa: BLE001
        reverted.append("G3")

    # --- G2: FLOAT -> BOOL (mask+margin gate) ---
    if allow_dtype:
        try:
            g2_model, g2_stats = dtype_opt.g2_float_to_bool(current)
            if g2_stats["converted_tensors"] > 0:
                _validate_pass(g2_model)
                margin_ok = scoring.masks_equal_with_margin(
                    current, g2_model, task_num, G2_MARGIN
                )
                cost_g2, _ = _cost_of(g2_model, task_num, workdir, "p002g2")
                cur_cost, _ = _cost_of(current, task_num, workdir, "p002g2pre")
                cost_ok = (
                    cost_g2 is not None
                    and cur_cost is not None
                    and cost_g2 < cur_cost
                )
                if margin_ok and cost_ok:
                    current = g2_model
                    applied.append("G2")
                else:
                    reverted.append("G2")
        except Exception:  # noqa: BLE001
            reverted.append("G2")

    # --- G1: selective FP16 (mask+margin gate, only on the 3 G1 tasks) ---
    if allow_dtype and task_num in G1_FP16_TASKS:
        try:
            g1_model, _g1_stats = dtype_opt.g1_fp16_convert(current)
            _validate_pass(g1_model)
            margin_ok = scoring.masks_equal_with_margin(
                current, g1_model, task_num, G2_MARGIN
            )
            cost_g1, _ = _cost_of(g1_model, task_num, workdir, "p002g1")
            cur_cost, _ = _cost_of(current, task_num, workdir, "p002g1pre")
            cost_ok = (
                cost_g1 is not None
                and cur_cost is not None
                and cost_g1 < cur_cost
            )
            if margin_ok and cost_ok:
                current = g1_model
                applied.append("G1")
            else:
                reverted.append("G1")
        except Exception:  # noqa: BLE001
            reverted.append("G1")

    changed = current is not winner
    if changed:
        cost_after, score_after = _cost_of(current, task_num, workdir, "p002final")
    else:
        cost_after, score_after = cost_before, score_before

    return P002Outcome(
        model=current,
        applied=applied,
        reverted=reverted,
        cost_before=cost_before,
        cost_after=cost_after,
        score_before=score_before,
        score_after=score_after,
        changed=changed,
    )


def _optimize_source(
    source: str, task_num: int, workdir: str
) -> CandidateResult | None:
    """Load, optimize, validate and score one source for one task.

    Returns a populated ``CandidateResult`` when the source file exists,
    otherwise ``None`` (missing source). ``correct`` reflects whether the
    optimized model passed full verification + scoring.
    """
    src_path = SOURCE_DIRS[source] / f"task{task_num:03d}.onnx"
    if not src_path.is_file():
        return None

    base_model = onnx.load(str(src_path))
    optimized, applied, reverted = _apply_pipeline(base_model, task_num)
    scored = scoring.score_and_verify(
        optimized, task_num, workdir, label=f"{source}opt"
    )
    if scored is None:
        return CandidateResult(
            source=source,
            correct=False,
            passes_applied=applied,
            passes_reverted=reverted,
            note="validation/scoring failed",
        )
    return CandidateResult(
        source=source,
        correct=True,
        memory=scored["memory"],
        params=scored["params"],
        cost=scored["cost"],
        score=scored["score"],
        model=optimized,
        passes_applied=applied,
        passes_reverted=reverted,
    )


def _resolve_divergent_source_a(
    task_num: int, workdir: str
) -> CandidateResult:
    """Resolve a task where baseline source A fails LOCAL validation.

    These tasks are known to pass the official Linux grader (the float
    ``> 0.0`` boundary diverges from macOS arm64). We never swap sources here.
    Instead we apply S1/S2/S4 (and S3 unless skipped) to source A only and
    accept the optimized model ONLY when it is bit-identical to the original A
    (``outputs_bit_identical``) AND is scorable with ``require_correct=False``
    (so a local gold mismatch does not reject it). If accepted, the optimized-A
    model is the winner; otherwise the original source-A file is kept unchanged.

    Always returns a ``CandidateResult`` for source ``A`` carrying the chosen
    path in its ``note``. ``model`` is set only when optimized-A is accepted
    (identity-verified); otherwise ``model`` is ``None`` and the caller copies
    the original A file.
    """
    src_path = SOURCE_DIRS[BASELINE_SOURCE] / f"task{task_num:03d}.onnx"
    if not src_path.is_file():
        return CandidateResult(
            source=BASELINE_SOURCE,
            correct=False,
            note="divergent-local: source A missing",
            is_baseline=True,
        )

    original_a = onnx.load(str(src_path))
    optimized_a, applied, reverted = _apply_pipeline(original_a, task_num)

    # Score with the gold-correctness requirement dropped: divergent tasks fail
    # the local gold match by construction. Other rejection reasons (memory /
    # params None or negative, file size, sanitize failure) still apply.
    scored = scoring.score_and_verify(
        optimized_a,
        task_num,
        workdir,
        label="Adiv",
        require_correct=False,
    )

    identity_ok = False
    if scored is not None:
        identity_ok = scoring.outputs_bit_identical(
            original_a, optimized_a, task_num
        )

    if scored is not None and identity_ok:
        return CandidateResult(
            source=BASELINE_SOURCE,
            correct=True,
            memory=scored["memory"],
            params=scored["params"],
            cost=scored["cost"],
            score=scored["score"],
            model=optimized_a,
            passes_applied=applied,
            passes_reverted=reverted,
            note="divergent-local: identity-verified vs original A",
        )

    # Not accepted: keep the ORIGINAL source-A file unchanged.
    return CandidateResult(
        source=BASELINE_SOURCE,
        correct=False,
        passes_applied=applied,
        passes_reverted=reverted,
        note="divergent-local: kept original A",
    )


def _score_baseline(task_num: int, workdir: str) -> CandidateResult | None:
    """Score the source-A original file as-is (no optimization passes)."""
    src_path = SOURCE_DIRS[BASELINE_SOURCE] / f"task{task_num:03d}.onnx"
    if not src_path.is_file():
        return None
    base_model = onnx.load(str(src_path))
    scored = scoring.score_and_verify(
        base_model, task_num, workdir, label="Abase"
    )
    if scored is None:
        return CandidateResult(
            source=BASELINE_SOURCE,
            correct=False,
            note="baseline A failed validation",
            is_baseline=True,
        )
    return CandidateResult(
        source=BASELINE_SOURCE,
        correct=True,
        memory=scored["memory"],
        params=scored["params"],
        cost=scored["cost"],
        score=scored["score"],
        is_baseline=True,
    )


def _record_002(result: TaskResult, outcome: P002Outcome | None) -> None:
    """Copy a ``P002Outcome`` (if any) into a ``TaskResult`` for reporting."""
    if outcome is None:
        return
    result.p002_applied = outcome.applied
    result.p002_reverted = outcome.reverted
    result.cost_pre_002 = outcome.cost_before
    result.cost_post_002 = outcome.cost_after
    result.score_pre_002 = outcome.score_before
    result.score_post_002 = outcome.score_after


def _finalize_winner_model(
    task_num: int,
    winner: onnx.ModelProto,
    workdir: str,
    passes: str,
    *,
    allow_dtype: bool,
) -> P002Outcome | None:
    """Apply 002 passes (when selected) to a winner model, then save it.

    For ``passes == '001'`` this just saves ``winner`` (byte-for-byte identical
    to the legacy behaviour). For ``passes == '002'`` it runs
    ``_apply_002_passes`` and saves the resulting (possibly unchanged) model.
    When nothing changed, the same ``winner`` object is saved, so the output is
    identical to what 001 would have written.
    """
    if passes == PASSES_001:
        _save_winner(task_num, winner)
        return None
    outcome = _apply_002_passes(
        winner, task_num, workdir, allow_dtype=allow_dtype
    )
    _save_winner(task_num, outcome.model)
    return outcome


def _finalize_copied_source_a(
    task_num: int, workdir: str, passes: str, *, allow_dtype: bool
) -> P002Outcome | None:
    """Finalize a task whose 001 winner is the ORIGINAL source-A file.

    Under ``001`` (or when 002 changes nothing) the original A file is copied
    byte-for-byte (``shutil.copyfile``), preserving exact bytes. Under ``002``,
    if a residual pass actually improves the model the modified model is saved
    instead; otherwise the original file is copied unchanged.
    """
    if passes == PASSES_001:
        _copy_source_a(task_num)
        return None
    src = SOURCE_DIRS[BASELINE_SOURCE] / f"task{task_num:03d}.onnx"
    if not src.is_file():
        return None
    winner = onnx.load(str(src))
    outcome = _apply_002_passes(
        winner, task_num, workdir, allow_dtype=allow_dtype
    )
    if outcome.changed:
        _save_winner(task_num, outcome.model)
    else:
        # No change: copy the original bytes verbatim (byte-identical to 001).
        _copy_source_a(task_num)
    return outcome


def _process_divergent_task(
    task_num: int, baseline: CandidateResult | None, workdir: str, passes: str
) -> TaskResult:
    """Handle a task whose baseline A fails LOCAL validation (no source swap).

    Applies S1/S2/S4 (+S3 unless skipped) to source A only and either saves the
    identity-verified optimized-A model or copies the original A file unchanged.
    Never considers sources B or C. Under ``--passes 002`` the residual G3 pass
    (bit-identity-gated) is additionally applied; G1/G2 are EXCLUDED on
    divergent-local tasks (``allow_dtype=False``) because only identity can be
    verified for them.
    """
    resolved = _resolve_divergent_source_a(task_num, workdir)

    p002: P002Outcome | None = None
    if resolved.correct and resolved.model is not None:
        p002 = _finalize_winner_model(
            task_num, resolved.model, workdir, passes, allow_dtype=False
        )
        chosen_source = "A(identity)"
        chosen = resolved
        warning = (
            "divergent-local: baseline A failed local validation; "
            "saved identity-verified optimized A"
        )
    else:
        p002 = _finalize_copied_source_a(
            task_num, workdir, passes, allow_dtype=False
        )
        chosen_source = "A(orig-divergent)"
        chosen = None
        warning = (
            "divergent-local: baseline A failed local validation; "
            "kept original A (optimization not identity-verified)"
        )

    result = TaskResult(
        task_num=task_num,
        chosen_source=chosen_source,
        baseline=baseline,
        chosen=chosen,
        candidates=[resolved],
        warning=warning,
        fallback_used=False,
        divergent_local=True,
    )
    _record_002(result, p002)
    return result


def process_task(
    task_num: int, sources: list[str], workdir: str, passes: str = DEFAULT_PASSES
) -> TaskResult:
    """Run the full per-task pipeline and pick the best correct candidate.

    Two regimes (Amendment 1):

    * Baseline source A validates locally → keep the original best-of-source
      behaviour (optimize every source, baseline A in the pool, lowest-cost
      correct candidate wins).
    * Baseline source A FAILS local validation → never swap sources. Keep
      source A: accept optimized-A only if it is bit-identical to the original
      A and scorable; otherwise copy the original A file unchanged.

    When ``passes == '002'`` the residual passes G3 (no-op removal), G2
    (FLOAT->BOOL) and G1 (selective FP16) are applied to the chosen winner with
    the per-task verification gates of proposal 002 section 3.
    """
    baseline = _score_baseline(task_num, workdir)

    baseline_divergent = (
        baseline is not None
        and not baseline.correct
        and (SOURCE_DIRS[BASELINE_SOURCE] / f"task{task_num:03d}.onnx").is_file()
    )
    if baseline_divergent:
        return _process_divergent_task(task_num, baseline, workdir, passes)

    candidates: list[CandidateResult] = []
    for source in sources:
        result = _optimize_source(source, task_num, workdir)
        if result is not None:
            candidates.append(result)

    valid = [c for c in candidates if c.correct and c.cost is not None]
    # Include the unoptimized baseline (source-A original) in the selection
    # pool so that a correct baseline can never be regressed by a more
    # expensive optimized B/C candidate. Acceptance: cost after <= baseline A.
    if baseline is not None and baseline.correct and baseline.cost is not None:
        valid = valid + [baseline]
    warning = ""
    fallback_used = False
    chosen: CandidateResult | None = None
    p002: P002Outcome | None = None

    if valid:
        # Deterministic tie-break: lowest cost first, then prefer optimized
        # candidates over the baseline at equal cost (so the report still
        # shows pass effects), then source order A < B < C.
        def _sort_key(c: CandidateResult) -> tuple[int, int, int]:
            source_order = sources.index(c.source) if c.source in sources else len(sources)
            return (c.cost, 1 if c.is_baseline else 0, source_order)

        chosen = min(valid, key=_sort_key)
        if chosen.is_baseline:
            # Baseline has no in-memory optimized model; its winner is the
            # original source-A file.
            p002 = _finalize_copied_source_a(
                task_num, workdir, passes, allow_dtype=True
            )
        else:
            winner_model = chosen.model
            if winner_model is not None:
                p002 = _finalize_winner_model(
                    task_num, winner_model, workdir, passes, allow_dtype=True
                )
    else:
        warning = "all candidates failed validation; copied original source-A file"
        fallback_used = True
        _copy_source_a(task_num)

    if baseline is not None and not baseline.correct and not warning:
        warning = "baseline A failed validation (comparison unavailable)"

    if chosen is None:
        chosen_source = None
    elif chosen.is_baseline:
        chosen_source = "A(orig)"
    else:
        chosen_source = chosen.source

    result = TaskResult(
        task_num=task_num,
        chosen_source=chosen_source,
        baseline=baseline,
        chosen=chosen,
        candidates=candidates,
        warning=warning,
        fallback_used=fallback_used,
    )
    _record_002(result, p002)
    return result


def _save_winner(task_num: int, model: onnx.ModelProto) -> None:
    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OPTIMIZED_DIR / f"task{task_num:03d}.onnx"))


def _copy_source_a(task_num: int) -> None:
    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)
    src = SOURCE_DIRS[BASELINE_SOURCE] / f"task{task_num:03d}.onnx"
    if src.is_file():
        shutil.copyfile(src, OPTIMIZED_DIR / f"task{task_num:03d}.onnx")


# --- Task selection -----------------------------------------------------------


def parse_tasks(spec: str) -> list[int]:
    """Parse ``--tasks`` spec: ``all``, ``1-5``, or ``7,12,300``."""
    spec = spec.strip()
    if spec == "all":
        return list(range(1, NUM_TASKS + 1))
    tasks: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            tasks.update(range(lo, hi + 1))
        else:
            tasks.add(int(part))
    return sorted(t for t in tasks if 1 <= t <= NUM_TASKS)


def parse_sources(spec: str) -> list[str]:
    sources = [s.strip().upper() for s in spec.split(",") if s.strip()]
    for s in sources:
        if s not in SOURCE_DIRS:
            raise ValueError(f"Unknown source '{s}'. Valid: {sorted(SOURCE_DIRS)}")
    return sources


# --- Entry point --------------------------------------------------------------


def _report_config() -> reporting.ReportConfig:
    """Build the static report configuration from module-level constants."""
    return reporting.ReportConfig(
        reports_dir=REPORTS_DIR,
        optimized_dir=OPTIMIZED_DIR,
        submission_zip=SUBMISSION_ZIP,
        skip_tasks_s3=sorted(SKIP_TASKS),
        g1_fp16_tasks=sorted(G1_FP16_TASKS),
        g2_margin=G2_MARGIN,
        passes_001=PASSES_001,
        passes_002=PASSES_002,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Behaviour-preserving ONNX cost-reduction pipeline."
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Task selection: 'all', a range like '1-5', or a list '7,12,300'.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated candidate sources (default A,B,C).",
    )
    parser.add_argument(
        "--passes",
        default=DEFAULT_PASSES,
        choices=[PASSES_001, PASSES_002],
        help=(
            "Pass set: '001' = S1,S2,S4,S3 (legacy); "
            "'002' = 001 plus G3, then G2, then G1 (G1 only on its 3 tasks). "
            f"Default '{DEFAULT_PASSES}'."
        ),
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Build artifacts/submission.zip with optimized files flat at root.",
    )
    args = parser.parse_args(argv)

    task_nums = parse_tasks(args.tasks)
    sources = parse_sources(args.sources)
    passes = args.passes
    if not task_nums:
        print("No tasks selected.")
        return 1

    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)
    results: list[TaskResult] = []

    with tempfile.TemporaryDirectory(prefix="neurogolf_scratch_") as workdir:
        total = len(task_nums)
        for idx, task_num in enumerate(task_nums, start=1):
            result = process_task(task_num, sources, workdir, passes)
            results.append(result)
            chosen = result.chosen
            before = (
                f"{result.baseline.cost}"
                if result.baseline and result.baseline.correct
                else "n/a"
            )
            # Surface the realized (post-002) cost when a residual pass applied.
            if result.p002_applied and result.cost_post_002 is not None:
                after = f"{result.cost_post_002}"
            else:
                after = f"{chosen.cost}" if chosen else "FALLBACK"
            src = result.chosen_source or "A(copy)"
            extra = (
                f" 002[{','.join(result.p002_applied)}]"
                if result.p002_applied
                else ""
            )
            print(
                f"[{idx}/{total}] task{task_num:03d} "
                f"src={src} cost {before} -> {after}{extra}"
                + (f"  WARN: {result.warning}" if result.warning else "")
            )

    cfg = _report_config()
    run_number = reporting.next_run_number(REPORTS_DIR)
    md_text, json_payload = reporting.build_reports(
        results, sources, run_number, passes, cfg
    )
    md_path, json_path = reporting.write_reports(
        md_text, json_payload, run_number, REPORTS_DIR
    )
    print(f"Report: {md_path}")
    print(f"JSON:   {json_path}")

    if args.zip:
        zip_path = reporting.build_submission_zip(
            task_nums, OPTIMIZED_DIR, SUBMISSION_ZIP
        )
        print(f"Submission zip: {zip_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
