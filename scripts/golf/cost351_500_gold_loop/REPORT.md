# Cost 351–500 gold-exact search (8012.23)

## Outcome

- **Accepted candidates:** 0
- **Gold-safe gain:** `+0.000000`
- Root `submission.zip`, `submission_base_8012.23.zip`, `all_scores.csv`, and
  `best_score.json` were not written by this lane.
- Authority SHA-256:
  `720ebf75d826945250e3c7d7ea11780a950d8d3038546e9c7595503277a1189f`.

## Scope

- The ledger lists 37 tasks at cost 351–500.
- 9 known-black/private-zero tasks were excluded:
  `048, 102, 112, 134, 168, 170, 222, 333, 377`.
- `task014` was also excluded from this lane because the ledger says cost 356
  while the immutable authority member profiles at actual cost 288.
- 27 eligible tasks were divided across three workers.

## Searches completed

### History/archive, exact simplification, and low-cost transfer

- 15,578 candidate encounters.
- 1,402 matching loose ONNX files and 10,045 matching ZIP members examined.
- All known-data screening used a 100% exact sign target with no runtime,
  nonfinite, shape, or small-positive tolerance.
- No candidate survived the complete admission gate.

Evidence: `worker_0.json`, `worker_1.json`, `worker_2.json`.

### task338 archived cost-334 candidate

- Visible known-data sign accuracy was 100%, versus authority cost 403.
- It was rejected on the **first case of both independent fresh-2000 streams**:
  one stream emitted 2 values and the other 1 value in forbidden `(0, 0.25)`.
- It is quarantined by SHA-256 and cannot become a finalist.

Evidence: `task338_fresh_reject.json`.

### Exact static-shape folding

- Generated 15 cumulative `Shape` / `ConstantOfShape` initializer-folding
  variants.
- Every variant inherited strict ONNX full-check shape inconsistencies from its
  authority graph, so all failed the required structure gate before admission.

Evidence: `static_fold_evidence.json`.

### task275 shared-rank reduction

- Tested all 6 rank-1/rank-2 subsets of the shared `S/T/W` contraction factors.
- All 6 models passed structural checking but failed official local gold exact
  matching, and were rejected before fresh testing.

Evidence: `task275_rank_evidence.json`.

## Admission rule

Finalists require all of the following:

1. Strictly lower measured cost than the 8012.23 authority member.
2. `try_candidate.py` file, full-check/static-shape, official train/test/arc-gen
   gold, margin, runtime, and score gates.
3. Two independent 2,000-case fresh streams at 100%, with zero errors,
   nonfinite outputs, shape mismatches, or values in `(0, 0.25)`.

No POLICY90-only candidate is admitted.
