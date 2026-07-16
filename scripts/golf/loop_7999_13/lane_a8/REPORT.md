# Lane A8 — exact 7999.13 safety wave

## Result

No safe strict winner was found for tasks 012, 019, 034, 035, 046, 117,
and 125. Projected gain is `0.0`. No root ZIP, CSV, ledger, score pointer, or
shared handcrafted artifact was modified.

The authority is the exact `submission_base_7999.13.zip` archive with SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.

## New task035 experiment

The baseline final decoder stores each feature as `[x, 7x mod 256]`. A new
candidate replaced the second lane with zero and folded the apparent affine
threshold into an ordinary QLinearConv int32 bias. This removes six 10-byte
Mul outputs and nominally scores `545 -> 493` (`memory 496 -> 436`, `params
49 -> 57`). It passes the full ONNX checker and strict shape inference and
runs without exceptions under both ORT optimization modes.

It is not correct: both modes produce `0/266` known-correct, `266` mismatches,
and `0` runtime errors. The reason is that uint8 multiplication wraps modulo
256, so the second lane is a nonlinear color code rather than a redundant
linear multiple.

An exact follow-up integer-MILP audit checked all 64 subsets of the six scaled
lanes (`left2`, four `core*b` lanes, and `right2`) over 94 unique known
pack/label states. It constrained QLinearConv weights to the legal effective
range `[-128,127]`, included an ordinary int32 bias, and explicitly required
padding to stay negative. No single lane can be removed; only the original
zero-removal structure is separable.

## Other targets

- task012: the exact member already has zero intermediate memory. The complete
  lane_c4 hard-separation search checked 351 smaller kernel alignments and found
  none feasible; the first feasible geometry is the incumbent cost-710 7x10
  grouped Conv.
- task019: the only cost-535 variant empties unused Split output `ca0`; ORT
  exits `-11`, so it is rejected as an error candidate.
- task034: no cheaper proper loose/archive model exists. The dominant 14x14
  shift table has exact matrix rank 7, so a factorization does not reduce its
  196 parameters and would add charged runtime tensors.
- task035: 34 unique proper loose models have no static floor below 545. The
  only duplicate constants are rank-incompatible zeros, giving a cost tie.
- task046: 58 unique proper loose models have no static floor below 631; the
  exact member was also unchanged in lane_c4.
- task117: the exact member was unchanged in lane_c4. The apparent lower loose
  candidate relies on declared/runtime shape cloaking, prohibited by this lane.
- task125: lane_c3 already screened 85 unique models from 921 observations.
  Best alternate known-correct actual cost is 1051 versus incumbent 1050; the
  actual cost-321 alternate is known-incorrect. Lower static figures are
  CenterCropPad/value-info cloak artifacts and are excluded.

Fresh-5000 was not run because no candidate passed the complete known set; the
mandatory gate order stops at known failure.

Machine evidence is in `failure_manifest.json`, `loose_static_screen.json`,
`task035_fold_search.json`, and `profile_task035.json`.
