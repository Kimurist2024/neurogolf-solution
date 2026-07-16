# 8018.91 urgent lane B — cost 250–399

## Result

- **Strict winner: 0**
- **Gold-safe projected gain: +0.000000**
- Immutable comparison authority:
  `submission_base_8018.91.zip`
- Authority SHA-256:
  `e43865760ec8807fbb217fba718226ca6b86d9128b911479214e3252b9f9e091`
- The lane did not write `submission.zip`, `all_scores.csv`, `best_score.json`,
  or the authority ZIP.

Admission required official/local gold exactness, full checker and strict
static shape inference, stable margin, zero runtime/nonfinite/shape errors,
and two independent 2,000-case fresh streams at 100% in all four ORT
configurations.  No POLICY90 candidate was admitted.

## Scope

The current ledger contains 45 tasks at cost 250–399.  Seven maintained
private-zero/unsound tasks in the band were excluded (`048`, `168`, `170`,
`178`, `185`, `192`, `222`), leaving 38 tasks.

Three disjoint history workers searched loose ONNX history, ZIP history,
current exact simplifications, and low-cost graph transfers.  Workers 0 and 1
completed with 8,322 and 7,683 candidate encounters respectively and no
survivor.  Worker 2 completed all discovery phases and produced exactly one
known-gold lower survivor:

- `task338`, cost 382 -> 334, SHA-256
  `d7ea232e2e894d3749f4aebf67de754b29cba35f940147b9db314a59043781fe`.

This is byte-identical to the already quarantined archive candidate in
`cost351_500_gold_loop/task338_fresh_reject.json`.  Both independent
fresh-2000 streams fail on their first evaluated case with forbidden
small-positive values (2 and 1 elements respectively); earlier full fresh
analysis also found real sign errors.  Its redundant long full-audit rerun was
stopped and the prior same-SHA rejection was reused.  It is not a candidate.

## Exact initializer and graph scans

`scan_exact_dedup.py` tested exact duplicate initializer/Constant aliasing,
Constant-to-initializer materialization, neutral-node removal, static
Shape/Size folding, and ConstantOfShape folding.  It generated 22 unique
lower-graph variants.  All 22 inherited a full-check/strict-shape or truthful
output-shape failure from their source graph, so none reached gold/fresh
admission.

`scan_einsum_factor.py` ran exact serialized outer-product factorization with
global factor deduplication over all 38 eligible authorities.  Six tasks had a
factorable initializer, but the globally best projected parameter delta was
`-1`; no strict-lower graph exists in this family.

`scan_lowrank_census.py` examined 84 Einsum nodes, 133 constant operands, and
223 exact rational axis bipartitions.  Four R>=2 parameter-saving partitions
exist, all for `task398/K`; every one lacks enough independent legal latent
labels in the repeated Einsum uses.  Buildable structural candidates: 0.

## Evidence

- `worker_0.json`, `worker_1.json`: completed disjoint history workers.
- `exact_dedup_report.json`: 22 exact rewrite attempts and rejection reasons.
- `einsum_factor_report.json`: exact outer-factor/dedup optimization proof.
- `lowrank_census.json`: exact-rational R>=2 factor census.
- `scripts/golf/cost351_500_gold_loop/task338_fresh_reject.json`: same-SHA
  task338 fresh-2000x2 rejection.

During the lane, the mutable root `submission.zip` changed concurrently from
the authority SHA to `0d9cb04a90d9c8d2ae06dc5d3797956525babf06ccb9ce15325ef8fdb237ebaf`.
This lane continued exclusively against the pinned immutable 8018.91 base and
did not overwrite or revert the concurrent root update.
