# Wave B4 — exact factor/reuse result

One strict winner was found: `task107`, cost **744 → 708**, for a projected
score gain of **+0.04959694113937019**.  The accepted file is:

`scripts/golf/loop_7999_13/lane_b4/candidate_task107_shared_coefficients_rank4.onnx`

No root submission, score file, or leaderboard artifact was modified.

## Why task107 is exact

The incumbent's `Acoef` and `Bcoef` initializers contain 128 float16 elements.
Their first-axis slices have one exact duplicate (`Acoef[1] == Bcoef[0]`), so
the two tensors can first be represented by a three-slice shared bank plus two
one-hot selectors.  The resulting `3x2x2x2x2x2` bank has, when reshaped to
`12x8`, nonzero columns only at indices `3,5,6,7`.  It therefore factors
exactly into a `12x4` value bank and a `4x8` one-hot support tensor.  The final
representation uses 92 elements, saving 36 parameters, and adds no runtime
node.  The builder asserts exact initializer reconstruction before saving.

The stronger runtime proof is the independent 3000-case domain differential:
baseline and candidate raw tensors were bitwise identical on all 3000 cases
(`max_abs_raw_difference=0.0`), with zero baseline, candidate, or candidate-only
runtime errors.

## task107 gates

| Gate | Evidence |
|---|---|
| Exact baseline | ZIP SHA-256 `a123cdc6...a2e1`; member SHA-256 `a7b3bfe7...74664` |
| External known | 266/266 right, 0 wrong, 0 errors |
| Official + local gold | both true |
| Fresh generator | 5000/5000, 0 failures, threshold 100% |
| Independent differential | 3000/3000 gold; decoded and raw bitwise equal |
| Margin | stable; minimum 0.405029296875 |
| Cost | memory 470 + parameters 238 = 708 |
| Structure | checker/full inference pass; static positive; standard domains |
| UB/error exclusions | no banned/Sequence/nested/function/sparse/nonfinite/bias issue |
| File limit | 5029 bytes (<1.44 MB) |

The incumbent already uses one 50-input monolithic `Einsum`; the factorized
candidate has 58 inputs.  This is disclosed rather than hidden.  It adds no
node or op/domain, ran all 5000 fresh cases, and ran both sessions over a further
3000 cases without any runtime error or output bit change.  The stale declared
`1x10x1x1` output annotation is byte-for-byte inherited in meaning from the
incumbent; both models execute as `1x10x30x30`, and full checker/strict shape
inference accept both.

## Fallback

`candidate_task107_shared_coefficients.onnx` is a simpler exact factorization
at cost 724 (+0.027249642447372935).  It independently passed 266/266 known and
5000/5000 fresh, but is retained only as a fallback because cost 708 dominates
it.

## Other assigned tasks

- `task156` (556): its final quantized kernel has output rank 3, but an exact
  two-stage convolution would allocate at least 2700 scored runtime bytes to
  save only 96 parameters.
- `task251` (755): both relevant kernel/channel matricizations are full rank 4;
  no parameter-only exact factorization or repeated initializer exists.
- `task275` (432): `MA != BA` and `MB != BB`; trimming their shared zero tails
  requires a dynamic grid slice or padded runtime masks.  Historical cost-317
  polynomial models are known-gold wrong.
- `task310` (566): the 30-row `E3/E4` constants are shared by output and capped
  contraction axes, so the eight-row cap support cannot be trimmed without
  duplicating more parameters or adding more runtime memory than it saves.
- `task328` (558): `CoreB` has full mode ranks `(4,4,5)` and `TFeat` rank 4.
  The exploratory sum-CoreB model failed the first archived known case and was
  rejected before fresh validation.
- `task333` (449): `C5/C2` are prefixes of `D5/D2`, but direct reuse fails shape
  inference (10 versus 30).  Correct Slice-based reuse has a cost lower bound
  of 659 because its 70 float elements require 280 scored runtime bytes.

Machine-readable evidence is in `task107_rank4_known.json`,
`task107_rank4_verify_fix_5000.json`,
`task107_rank4_domain_differential.json`,
`task107_rank4_structural.json`, and `winner_manifest.json`.
