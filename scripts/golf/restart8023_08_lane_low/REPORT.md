# 8023.08 low-cost lane

## Decision

No additional model from this lane satisfies the current fail-closed promotion
gate.  The immutable authority and root submission files were not modified.

## Authority

- `submission_base_8023.08.zip`
- SHA-256: `0e29e8d57f7ac58136a9574351c9c6f3056f9debf6eeee9c181c8f2e9fac690a`
- Reprofiled updated members: task062, 156, 238, 297, 324, 341, 398.
- task175@131 was already staged by the root lane and was not duplicated here.

## Exact optimizer scan

The standard semantics-preserving ONNX optimizer family produced no valid
strict-lower candidate.  Several normalized variants returned profiler sentinel
`(-1,-1,-1)`; these are invalid profiles, not cost wins.  The scanner now
requires every profile component to be nonnegative plus full-check, strict
data-propagating inference, and static inferred outputs before it can label a
variant strict-lower.

The newly accepted low-cost authority members themselves inherit metadata
shortcuts:

- task156: runtime-shape mismatch at `g`.
- task238, 297, 324, 341: noncanonical one-channel graph-output declarations.
- task398: runtime-shape mismatches at `coord`, `mask_b`, and `mask`.
- task062: strict data-propagating inference rejects stale Reshape declarations.

These findings were not copied into a new promotion.

## Exact algebra candidate, rejected

`candidates/task324_outscale_reuse.onnx` removes two parameters by replacing
the two `[0,2]` operands in the final Einsum with the already-live `[0,1]`
operand and multiplying the same contraction by the exact element-sum four of
the already-live `signpow` tensor.

- authority cost: 425
- candidate cost: 423
- projected gain: `ln(425/423) = 0.004717`
- full checker: pass
- strict shape inference with `data_prop=True`: pass
- official/local gold: pass (`correct=true`)
- fresh seed 824324001: **reject**
  - 1999/2000 correct
  - one wrong case, 37 wrong cells
  - nonfinite values in all 2000 cases (721493 elements)
  - 39 positive elements below the required 0.25 margin
  - ORT_ENABLE_ALL configurations failed

The second fresh seed was cancelled after this decisive hard-gate failure.

## Sparse representation experiment

Dense-to-sparse conversions could save parameters on several members, but
ONNX strict inference exposes sparse tensor types to ordinary tensor operators
and rejects them.  No sparse candidate passed the required full-check plus
strict inference path; none was admitted.

## Artifacts

- `scan_exact_optimizers.py`
- `optimizer_scan.json`
- `outer_factor_scan.json`
- `basis_reuse_scan.json`
- `build_task324_outscale_reuse.py`
- `audit_task324_outscale_reuse.py`
- `build_sparse_exact.py`
- `sparse_exact_build.json`

