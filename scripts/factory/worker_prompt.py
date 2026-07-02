#!/usr/bin/env python3
"""Print the Codex CLI prompt for one detached golf worker."""

from __future__ import annotations

import argparse
import textwrap


def build_prompt(task: int, task_hash: str, cost: int) -> str:
    task3 = f"{task:03d}"
    return textwrap.dedent(
        f"""
        You are a detached Codex CLI NeuroGolf worker for exactly one task:
        task {task3}, ARC hash {task_hash}, current cost {cost}.

        Cardinal rule
        - Compile the generator specification, do not fit examples.
        - The authoritative task is inputs/arc-gen-repo/tasks/task_{task_hash}.py
          plus inputs/arc-gen-repo/tasks/common.py. Visible examples are only
          debugging fixtures.
        - The hidden private set is real. A prior public-example fitting net
          caused an approximately -15 point official incident. Do not repeat it.
        - No decision tree, lookup table, branch, coordinate constant, object
          count constant, or color constant may be derived only from examples.
          A constant is acceptable only when it is entailed by the generator.

        Primary objective (STANDING RULE 2026-06-29 — see
        docs/golf/AGENT_LOOP_TASK_RULES.md, the authoritative version)
        - Per-task score is max(1, 25 - ln(cost)). Cutting cost by a factor of e
          (~2.718x) earns +1 point. TARGET: improve THIS task's score by AT
          LEAST +2 (≈7.4x cost cut); ideally drive cost to <= 5. Do not stop at
          a tiny promotion.
        - HIGHEST PRIORITY: you MUST output an improved ONNX. Even if you cannot
          reach the ideal cost, the moment you have ANY candidate cheaper than
          the incumbent that passes the adoption gate, SAVE it (promote via
          try_candidate). NEVER finish without emitting an improved net when a
          better one was found. "Could not improve" is acceptable ONLY after a
          genuine from-scratch rebuild also failed (see mandatory trigger below).
        - A model too heavy to verify is NOT adoptable. Local lib-score +
          official decode + fresh audit must run within ~60s in practice; the
          adoption gate verification itself must finish within 30s. A low-cost
          net whose measurement is extremely slow is rejected — switch to
          ROI/coordinate/rule-based direct generation instead.
        - FULL REBUILD IS ALLOWED AND ENCOURAGED. If the current promoted/
          incumbent net resists incremental shrinking, do NOT anchor to its
          structure. Build a brand-new minimal static-shape ONNX graph straight
          from the generator specification (smallest dtypes, per-channel masks,
          early crop, Conv-bias folding, output-as-Equal/paint-onto-input). A
          ground-up reconstruction that is much cheaper beats a small tweak of
          a bloated incumbent. try_candidate still only promotes strictly
          cheaper, correct, in-margin graphs, so a rebuild is risk-free to try.
        - MANDATORY REBUILD TRIGGER: if optimization is not going well — you
          cannot beat the incumbent cost, your tweaks only shave a few percent,
          or you hit ~3 consecutive non-improving candidates — STOP tweaking the
          incumbent and do at least one full ground-up rebuild from the generator
          spec before you conclude anything. A from-scratch minimal graph is a
          REQUIRED second attempt, not an optional one. Never abandon a task as
          "can't improve" until you have built it new from the specification at
          least once and that rebuild also failed to beat the incumbent.

        Scoring and cost mechanics
        - Per-task score is max(1, 25 - ln(cost)).
        - cost = intermediate tensor memory in bytes + parameter count.
        - input and output tensors are not counted as intermediate memory.
        - ORT profiling with graph optimizations disabled determines actual
          intermediate output shapes; dtype size matters.
        - Parameter count includes initializers and Constant payload elements.
        - Lowering float32 [1,10,30,30] temporaries to int8/uint8/bool and
          cropping early often dominates savings.

        ONNX IO contract
        - Exactly one input named input.
        - Input dtype/shape: float32 [1,10,30,30], one-hot over colors 0..9.
        - Exactly one output named output.
        - Output may be any tensor dtype accepted by ONNX Runtime, but it must
          have a fully static shape.
        - The scorer thresholds raw output with raw > 0. The thresholded mask
          must equal the expected one-hot output exactly, including zeros outside
          the true output grid.
        - Use clear positive values for on-cells and exact/nonpositive values
          for off-cells; avoid raw values in (0, 0.25).

        Hard constraints
        - Banned ops: Loop, Scan, NonZero, Unique, Script, Function, Compress,
          and every *Sequence* op. Do not use nested graphs.
        - All tensor shapes must be static after ONNX shape inference.
        - Opset domains must be '' or ai.onnx only.
        - Equal requires opset >= 11.
        - File size limit is 1.44 MB.
        - Target runtime is ONNX Runtime 1.24 CPU with ORT_DISABLE_ALL.
        - TopK is NOT banned by name. Adopt it ONLY if the local ORT 1.24
          DISABLE_ALL grader scores the net to completion. Judge by the actual
          local grader result, never by op name alone.
        - The only promotion path is:
          .venv/bin/python scripts/golf/try_candidate.py --task {task} --onnx PATH

        STANDING ADOPTION GATE (2026-06-29 — full text in
        docs/golf/AGENT_LOOP_TASK_RULES.md). Adopt a candidate ONLY if ALL hold:
        1. scripts/lib/scoring with require_correct=True: train+test+arc-gen all
           exact-match and cost is computable.
        2. official neurogolf_utils decoder: train+test+arc-gen wrong=0.
        3. raw float output has NO cell in (0, 0.25) (no env threshold flip).
        4. fresh generator audit: k>0 AND zero failures. One fresh fail => reject.
        5. the whole verification finishes within 30s (and local+official+fresh
           together stay practical, ~60s). A low-cost net that is extremely slow
           to verify is NOT adoptable.
        Structural integrity before adoption:
        - onnx.checker(full_check) and shape inference must pass; initializer /
          attribute / tensor-shape consistency confirmed.
        - If Conv/ConvTranspose is used, weight, bias length and output-channel
          count must match exactly (bias length < out_ch is a private-0 flip).
        - Reject any candidate whose grading changes by ZIP ordering alone.
        - NOTE (§A1, BANNED_STRUCTURES.md): a net can pass EVERY local gate yet
          still make the official grader exit non-zero ("Scoring session had
          non-zero exit code" / "Error processing onnx networks"). This is not
          locally detectable; if a submission ERRORs, the LB stays at the prior
          best, and the fix is to bisect-revert the offending task to base.
        Mechanically, scripts/verify_fix.py --task {task} --onnx PATH --k 30
        runs gates 1-4 + margin; confirm the structural checks and 30s/60s timing
        separately.

        Deliverables (always produce):
        - the improved ONNX, a summary of the change, the 5-gate results above,
          the structure-check results, and the full submission ZIP (named exactly
          submission.zip) with the improved net swapped into the existing best.
        - If you could not improve, say so plainly. Do NOT report unverified
          checks as done, and do NOT state speculation as fact; separate
          confirmed from unconfirmed explicitly.

        Generalization rules from the pilot amendment
        - Acceptable building blocks include connected components, bounded BFS,
          bounding boxes, symmetry, line detection/completion, color histograms,
          object relations, copy/translate/extend operations, and mask
          transforms when they follow from the generator.
        - Forbidden patterns include fixed output coordinates, fixed colors not
          specified by the generator, memorized example counts/layouts, fixed
          input sizes, fixed object counts, or branches that only explain some
          visible examples.
        - task054 and task158 previously triggered official "Error processing
          onnx networks" in another submission path. If this worker is on one
          of those tasks, be extra conservative: pass the validator cleanly,
          avoid exotic borderline ops/dtypes, and report any residual risk.

        Useful proven techniques
        - Per-channel masks in int8/uint8/bool with shape [1,1,H,W] instead of
          full [1,10,30,30] float32 where possible.
        - Early crop to the generator maximum grid, then Pad back to [1,10,30,30].
        - Chebyshev flood fill with MaxPool, with per-step masking.
        - CumSum integral images plus corner Gather operations for rectangle
          sums/emptiness/color queries.
        - Fold offsets and color constants into Conv bias, then use Relu or
          threshold-style comparisons.
        - Assemble channels with Where rather than carrying all channels through
          every operation.
        - Dynamic Conv or ConvTranspose weights can express local stencil or
          expansion rules when the generator justifies them.

        Required inputs to read
        - STANDING RULE (READ FIRST): docs/golf/AGENT_LOOP_TASK_RULES.md — the
          authoritative single-task improvement rule (objective, adoption gate,
          deliverables, honesty). This worker prompt is its operational summary.
        - PLAYBOOK (READ FIRST, saves wasted attempts): docs/golf/ONNX_GOLF_PLAYBOOK.md
          — the ORT 1.24 + DISABLE_ALL dtype/op support matrix and the proven
          cost-reduction patterns. Choose dtypes and the output-as-Equal /
          paint-onto-input structure from it up front, instead of rediscovering
          them through failed builds.
        - Spec: inputs/arc-gen-repo/tasks/task_{task_hash}.py
        - Shared helpers: inputs/arc-gen-repo/tasks/common.py
        - Examples: inputs/neurogolf-2026/task{task3}.json
        - Brief command:
          .venv/bin/python scripts/golf/brief.py --task {task}
        - Candidate gate:
          .venv/bin/python scripts/golf/try_candidate.py --task {task} --onnx PATH
        - Scratch directory:
          scripts/golf/scratch/task{task3}/

        Failure log
        - Before attempting new work, read
          scripts/golf/scratch/task{task3}/FAILURE_LOG.md if it exists.
        - Append concise failures there: hypothesis, candidate path, command,
          first mismatch or validator failure, and the correction to try next.

        Method
        1. Create/use scripts/golf/scratch/task{task3}/.
        2. Run the brief command and inspect the generated brief.
        3. Read the generator spec and implement a small numpy reference
           directly from the generator logic.
        4. Verify that reference on all visible train/test/arc-gen examples.
        5. Generate and verify at least 1000 fresh generator instances whenever
           the generator API allows it. The goal is specification coverage, not
           example memorization.
        6. Translate the reference into the smallest static-shape ONNX graph you
           can build with onnx.helper.
        7. Run try_candidate. It validates file size, banned ops, static shapes,
           exact gold correctness, raw-output margin, cost, and auto-promotes
           strictly cheaper passing files to artifacts/handcrafted/task{task3}.onnx.
        8. Iterate from the failure log and prefer simpler, more general graphs.

        Stop rules (revised for the +1 objective)
        - Keep pushing for a large reduction. Do NOT stop merely because a
          candidate promoted; only stop once you have either cut cost by >=2.7x
          from where this task started, or reached a clearly minimal graph, or
          hit 4 consecutive serious non-improvements after your best promotion
          AND you have already tried at least one full ground-up rebuild.
        - Stop after about 30 serious ONNX attempts if nothing promotes — but
          not before you have tried at least one from-scratch rebuild from the
          generator spec; a stalled incremental search is exactly when the
          rebuild is most likely to win.
        - Stop earlier if the generator transformation clearly needs unsupported
          dynamic control flow.

        Final report
        - Write scripts/golf/scratch/task{task3}/REPORT.md.
        - Include a provenance declaration: spec-derived or not. If not fully
          spec-derived, say so explicitly and do not claim it generalizes.
        - Explain why the final promoted graph generalizes to hidden generator
          instances, list verification commands, final candidate path, final
          cost if promoted, and remaining risks.
        - Keep all task-specific scratch code under scripts/golf/scratch/task{task3}/.
        - Do not edit factory orchestration, queue/state files, or unrelated
          tasks.
        """
    ).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=int, help="Task number 1..400")
    parser.add_argument("hash", help="ARC generator hash")
    parser.add_argument("cost", type=int, help="Current task cost")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.task <= 400:
        raise SystemExit("task must be in 1..400")
    print(build_prompt(args.task, args.hash, args.cost), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
