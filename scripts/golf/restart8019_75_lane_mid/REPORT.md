# 8019.75 mid-cost evidence lane

## Authority and safety boundary

- Immutable authority: `submission_base_8019.75.zip`
- SHA-256: `e69058edd21e27ab7d32670d714ec5cea6d35632a9d9a620364731297717edb3`
- Scope: ledger cost 250 through 399, plus a narrow reopened high-risk pass.
- Admission: official/local gold exact, full checker and strict/static shape pass,
  stable raw margin, fresh 2,000 x 2 at 100% in four ORT configurations, and
  zero runtime errors/nonfinite outputs/shape mismatches/small-positive outputs.
- This lane did not update `submission.zip`, `all_scores.csv`, or
  `best_score.json`.
- `task338@334` remains explicitly banned.  Its old SHA was observed by the
  first launch of worker 2 and that process was stopped before admission; the
  wrapper now permanently adds task338 to the exclusion set.

## Strict winner

| task | authority | candidate | gain | SHA-256 |
|---|---:|---:|---:|---|
| 345 | 389 | 369 | +0.0527826996 | `1b6b180284a61b4a734137c4e43d6fc1f928f4e5179a271563267d3247dfafdf` |

Candidate:
`candidates/task345_POLICY90_cost369_1b6b180284a6.onnx`

Evidence in `worker_1.json`:

- official/local gold: 264/264 exact;
- cost profile: memory 308, params 61, cost 369;
- full checker and scanner strict/static gate: pass;
- fresh seeds 408110345 and 408210345: 2,000/2,000 exact each in
  ORT_DISABLE_ALL/ORT_ENABLE_ALL at 1 and 4 threads;
- every fresh configuration has errors 0, nonfinite 0, runtime output shape
  mismatches 0, small-positive outputs 0, and raw/sign stability;
- independent non-mutating verifier: `correct=true`, `cost=369`.

The verifier emits two ORT shape-merge warnings for undeclared PRelu/CastLike
intermediates, while the model full checker, strict scanner, declared output
shape, and runtime output shape all pass.  This warning is retained here for
the central admission decision rather than hidden.

At lane completion the mutable `submission.zip` already contained this exact
task345 member SHA, while `all_scores.csv` still recorded task345 cost 389.
That change came from another lane; this lane only copied the member into its
candidate directory and audited it against the immutable 8019.75 authority.

## Other scans

- History/exact/transfer worker 0: 13 assigned tasks, no finalists.
- History/exact/transfer worker 1: 13 assigned tasks, only task345 above.
- Exact serialized dedup scan: 30 variants, all 30 preflight rejected, no
  finalist.
- Exact Einsum outer-factor scan: 46 tasks, no finalist.
- Low-rank census: 46 tasks, 93 Einsum nodes, 153 constant operands, 238
  partitions.  Four rank-2 parameter-saving partitions exist only on task398,
  but every one lacks enough unused Einsum labels, so there is no structural
  candidate.
- Reopened worker 0 (tasks 48/178/222): no finalist.  task48@142 passed local
  examples but failed fresh at 64.35% and 62.10%.
- Reopened worker 2 (tasks 170/192): no survivor/finalist.
- Reopened worker 1 (tasks 168/185/354): nine local-gold survivors in known
  LB-black/private-zero families were quarantined before admission; task354
  yielded no survivor.  Details are in `reopened/worker_1_quarantine.json`.

## task345 micro follow-up

Replacing its PRelu factor reconstruction with Mul, or changing the factors to
float32 and removing CastLike, stayed official-gold exact but increased cost to
709 or 909.  Those models under `task345_micro/` are rejected experiments, not
candidates.

## Projected score

Against the requested 8019.75 authority, the sole strict-gate improvement is
task345's +0.0527826996, for a local projection of approximately **8019.8028**.
