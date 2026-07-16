# task319 Log-scale Selu screen

## Outcome

No candidate is admissible. The proposed removal of the single-use
`inv_ln2_f32` initializer is not exact on generator-valid inputs, so the
candidate was not built or staged. Projected gain is `+0.0`.

## Decisive support trace

`trace_source.py` exposed the authority tensors immediately before and after
the relevant `Log` and ran 10,000 independent generator cases. All 10,000
executed without an error.

- `max_abs_f16` minimum: `0.5`
- `max_abs_f16` zero/nonfinite count: `0 / 0`
- `log_abs_f32` minimum: `-0.693359375`
- negative `log_abs_f32` cases: `113`
- zero `log_abs_f32` cases: `27`
- nonfinite `log_abs_f32` count: `0`

The authority computes a linear positive scaling of `log_abs_f32`. Replacing
that multiplication with `Selu(gamma=inv_ln2)` is linear only on the
nonnegative branch; for the reachable negative values Selu applies its
exponential branch. Therefore the rewrite cannot be authority-bitwise or
threshold exact on valid support. Because task319 is a private/non-injective
lineage, sampled accuracy cannot substitute for the required pass-through
guarantee.

Root `submission.zip`, `all_scores.csv`, and `others/71407` were not modified.
