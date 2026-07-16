# Lane B9 report

No candidate was accepted for the exact `submission_base_7999.13.zip` base.
The strict gain for this lane is therefore `+0.0`, and no root ZIP, CSV, or
ledger file was changed.

## Exact baselines

The base ZIP SHA-256 is
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
The audited task costs were:

| task | cost | memory | params |
|---:|---:|---:|---:|
| 156 | 556 | 330 | 226 |
| 182 | 994 | 930 | 64 |
| 216 | 1511 | 1465 | 46 |
| 237 | 529 | 413 | 116 |
| 238 | 562 | 418 | 144 |
| 284 | 518 | 465 | 53 |
| 379 | 1955 | 1570 | 385 |

All seven exact members passed full ONNX checking and strict shape inference,
use only the standard ONNX domain, and contain no functions or sparse
initializers.  The detailed inventory is in `baseline_inventory.json`.

## History scan

The ZIP sweep's 20 retained references and 72 unique loose/local candidates
that survived the static prefilter were profiled with the real scorer.  None
was strictly cheaper than its exact baseline.  The best archive costs were
task182 `1062 > 994`, task216 `1525 > 1511`, and task284 `521 > 518`.

Evidence:

- `archive_actual_scores.json`
- `local_history_inventory.json`
- `local_history_actual_scores.json`

## task182 constant-reuse lead (rejected)

`task182_reuse_constants.onnx` reduced parameters from 64 to 60 while leaving
the measured memory at 930, for a nominal cost improvement `994 -> 990`.
It made four algebraically exact changes:

1. `s5` became `Add(s2, s3)`.
2. `s19` became `Sub(Mul(s4, sh5), s1)` while retaining the same two runtime
   shape tensors as the base.
3. `Mul(count_i32, -1)` became `Neg(count_i32)`.
4. `Mul(col_count, 6)` became `Selu(col_count, gamma=6)`.  `col_count` is a
   sum of nonnegative one-hot entries, so SELU's nonnegative branch is exactly
   `6*x`, including zero.

The candidate passed known examples, full checker, strict inference, standard
domain, safe QLinearConv bias shape, and generator fresh `5000/5000` under
`ORT_DISABLE_ALL`.  It was nevertheless rejected because default ORT could
not construct a session: constant folding exposes the base model's inherited
scalar shape for a two-axis `CenterCropPad`.

The failure is recorded in `task182_dual_ort_5000.json`; structural and cost
evidence is in `task182_candidate_audit.json`.

## Honest shape experiment

`task182_static_shapes.onnx` replaced the shape chain with axes-length-matched
initializers and removed the false intermediate value-info declarations.  It
is correct, but its real cost is `169429`, so it cannot beat 994.  This confirms
that using a new shape/value-info cloak would be required to retain the low
reported memory; that route was intentionally rejected.

## Initializer/CSE audit conclusion

No exact duplicate initializer or exact duplicate expression exists in any of
the seven baselines.  The remaining apparent table overlaps require a runtime
materialization whose memory exceeds the saved initializer elements (notably
task379 `E/E0/coord/vzero`, task238 position/route tables, and task284
`x3/x4`).  Dense-kernel factorization for tasks 156 and 237 likewise introduces
spatial intermediates larger than the parameter reduction.  No such candidate
was advanced to the strict gate.
