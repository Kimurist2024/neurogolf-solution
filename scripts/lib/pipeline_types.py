"""Shared dataclasses for the NeuroGolf optimization pipeline.

Extracted from ``optimize_submission.py`` to keep that module under the
800-line budget. These are plain data carriers used by the pipeline driver and
the reporting module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import onnx

# Default G1/G2 mask-equality margin (proposal 002 section 3). Mirrored in
# ``optimize_submission.G2_MARGIN``; kept as a literal here to avoid a circular
# import between the driver and this types module.
DEFAULT_MARGIN: float = 0.25


@dataclass
class CandidateResult:
    """Outcome of optimizing one source for one task."""

    source: str
    correct: bool
    memory: int | None = None
    params: int | None = None
    cost: int | None = None
    score: float | None = None
    model: onnx.ModelProto | None = None
    passes_applied: list[str] = field(default_factory=list)
    passes_reverted: list[str] = field(default_factory=list)
    note: str = ""
    is_baseline: bool = False


@dataclass
class TaskResult:
    """Aggregated result for one task across all candidate sources."""

    task_num: int
    chosen_source: str | None
    baseline: CandidateResult | None
    chosen: CandidateResult | None
    candidates: list[CandidateResult]
    warning: str = ""
    fallback_used: bool = False
    # Set when the baseline A fails LOCAL validation but is known to pass the
    # official grader (float ``> 0.0`` boundary divergence). On these tasks we
    # never swap sources; we keep source A (optimized only when bit-identical to
    # the original) so the official score is not regressed.
    divergent_local: bool = False
    # Proposal 002 residual passes applied to the chosen winner (post-001).
    p002_applied: list[str] = field(default_factory=list)
    p002_reverted: list[str] = field(default_factory=list)
    # Cost/score of the winner BEFORE 002 passes and AFTER 002 passes.
    cost_pre_002: int | None = None
    cost_post_002: int | None = None
    score_pre_002: float | None = None
    score_post_002: float | None = None
    margin_used: float = DEFAULT_MARGIN


@dataclass
class P002Outcome:
    """Result of applying the proposal-002 residual passes to a winner model."""

    model: onnx.ModelProto
    applied: list[str]
    reverted: list[str]
    cost_before: int | None
    cost_after: int | None
    score_before: float | None
    score_after: float | None
    changed: bool
