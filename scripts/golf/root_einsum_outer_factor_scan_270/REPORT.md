# Einsum exact outer-factor scan (lane 270)

## Result

No strict-lower candidate exists for this transformation family.  Neither the
immutable root authority nor `others/71407` was modified.

## Scope

- Root authority: all 400 models from `submission.zip`, SHA-256
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
- Active descendants: all 19 files listed by `others/71407/MANIFEST.json`.
- Source models scanned: 419.
- Einsum nodes: 672 (authority 605, active descendants 67).
- Constant initializers whose every graph use is an Einsum operand: 611.
- Nontrivial unordered axis bipartitions checked: 1,221.
- Initializers with at least one exact rank-1 bipartition: 55
  (authority 49, active descendants 6).
- Exact serialized factor options: 22,923.

For tasks with an active descendant, that descendant is the cost baseline.  An
authority-derived candidate would only proceed if it beat the active model.

## Exactness method

For each rank-two-or-higher constant operand, every nontrivial axis
bipartition is visited exactly once by fixing axis zero in the first unordered
partition.  The permuted tensor is treated as a matrix.  Every 2x2 minor is
checked over the exact rational number encoded by the serialized integer,
float16, float32, or float64 value; no tolerance or SVD threshold is used.

Candidate factors must:

1. be exactly representable in the original initializer dtype;
2. reconstruct the original tensor using multiplication in that dtype; and
3. reproduce the original contiguous initializer bytes exactly, including
   signed-zero bits.

Nonfinite and complex constants are rejected.  Four synthetic checks cover
float32, float16, and int32 rank-1 tensors plus a non-rank-1 rejection.

An initializer can be removed only if every use is a compatible Einsum operand.
All uses would be rewritten together.  Factor accounting deduplicates identical
dtype/shape/byte tensors across all selected initializers and reuses compatible
existing initializers.

## Parameter lower bound

Only 39 source lineages contained one or more factorable initializers.  Their
globally optimized results were:

- 4 lineages: exact factor dedupe reaches equal parameter count;
- 35 lineages: the best exact factor set adds one parameter;
- 0 lineages: strict parameter reduction.

The four exact equal-cost cases are authority tasks 048, 157, 316, and 379.
Their respective best removed/added parameter counts are 1/1, 10/10, 60/60,
and 60/60.

To rule out an unenumerated factor scale enabling more dedupe, a second,
strictly optimistic DP ignores serialized scale compatibility: any same-dtype,
same-shape factor with the same exact rational direction is assumed freely
shareable, even if no common representable scaling exists.  This upper bound
still gives only six equal-cost and 33 worse lineages; its maximum possible
saving is zero.  Therefore no omitted scale can produce a strict-lower result.

All six factorable active-descendant initializers also have best saving `-1`;
none can improve its active 71407 baseline.

## Validation gates

Candidate count is zero, so no algebra-only rewrite was reported and no
operation-order-sensitive model reached profiling.  Full/strict/data-prop,
truthful runtime shape, known4 raw, UB0, and independent fresh 2 x 2000 gates
are intentionally not run without a strict-lower candidate.

Complete per-lineage proofs, factor hashes, exact DP results, and the optimistic
scale-free upper bound are in `scan_result.json`.

