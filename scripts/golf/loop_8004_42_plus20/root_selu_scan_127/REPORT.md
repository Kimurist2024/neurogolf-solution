# Root Selu exact memshave scan 127

## Result

Five strict-lower task candidates were admitted against the immutable
8009.46 authority.  `submission.zip`, `all_scores.csv`, and the authority ZIP
were not modified.

| task | official cost | candidate SHA-256 | projected gain |
|---:|---:|---|---:|
| 013 | 357 -> 356 | `97d6a181110e43e8a5b20031ac766bc38fa8d5787070a7bc026306a2da1c7173` | +0.002805050928 |
| 090 | 1050 -> 1049 | `32e234adc2f18b0487f08f50ba8c56a6190053534bccbe3539c3f554974cb695` | +0.000952834755 |
| 134 | 423 -> 422 | `a31b93b1d10cccbf7d07a3b68fca4250eb06d7f9bd44489f7fd8871ca8794f93` | +0.002366865010 |
| 209 | 2087 -> 2085 | `87690aaddd78db9a54a41b4a11edb73d503966eb8d27b4b60a3569fd1db0a751` | +0.000958772844 |
| 366 | 7987 -> 7985 | `8cdf40d19af0c539f11022e569fb11f7fdf03617775019bf1cfca31fb309b736` | +0.000250438268 |

Combined projected gain: `+0.007333961806`.

## Exact transformation

Each rewrite replaces every use of a positive float16 scalar initializer in
`Mul(x, g)` with `Selu(x, alpha=1, gamma=g)`, or `Div(x, d)` with
`Selu(x, alpha=1, gamma=1/d)`, then removes the unused one-element initializer.
Memory is unchanged and parameters decrease.

- task013: one use of `half_h`. Valid inputs contain two markers at long-axis
  coordinates `a,b`; the graph computes `P=a+b` and
  `d=sqrt(2(a^2+b^2)-(a+b)^2)=|a-b|`, hence `p0x2_h=P-d=2*min(a,b)>=0`.
- task134: three uses of `hInv29`. `rowq` and `col` decode nonnegative
  coordinates. `m2` adds a nonnegative even square-root term and an IDF
  vectorizer whose five weights are all `+2`.
- task209: two uses of `fhalf16` plus one use of `ln2`. `ysr16` and `ysc16`
  are casts of Einsum sums over nonnegative one-hot input, a 0/1 selector, and
  coordinates `0..29`; `pclowbit` is a positive power of two on valid inputs.
- task366: one use of `safe_name_19` and two uses of `safe_name_31`.
  `safe_name_40` is `Cast(Einsum(one-hot input, coordinates 0..29))`.
  `safe_name_752/753` cast sums of nonnegative coordinate gathers and bounded
  nonnegative uint8/int8 offsets. Two scalar parameters are removed.

The sources are finite, nonnegative, and cannot produce negative zero.  For
positive `x`, Selu's positive branch is `g*x`; for positive zero its negative
branch is also positive zero.

## Numerical completeness proof

`exhaust_float16.py` and `exhaust_div_float16.py` evaluated all 31,744
nonnegative finite binary16 bit patterns (including positive zero) for every
exact scalar. Under both ORT_DISABLE_ALL and default optimization, Mul/Div and
Selu were bitwise identical for every admitted task/scalar pair. The two Selu
runtime modes were also bitwise identical. See `exhaust_float16.json` and
`exhaust_div_float16.json`.

This exhaustive operator-domain result plus the source nonnegativity proof is
an all-valid-input pass-through guarantee, not a private-zero approximation.

## Runtime validation

- Full ONNX checker, strict shape inference with data propagation, and Conv
  bias UB checks pass for all five candidates.
- Known corpus: candidate and authority raw outputs are identical in four
  runtime configurations. Tasks013/090 cover267/267, tasks134/209 cover266/266
  converted cases per configuration; task366 covers all255 scorer-convertible
  cases, with the same
  11 oversized cases skipped by the scorer.
- Fresh generator: task013 is raw bitwise-identical on two streams totaling500
  cases per mode; task090 uses two500-case streams and the other tasks use
  two1,500-case streams. All have zero
  runtime errors. Task013 also has the earlier complete 37,800-state structural
  support proof for the current LB-white rule.
- Minimum fresh truth rate is task090 98.6%, task134 99.0%, task209 95.733%, task366 98.267%,
  above the user's 90% admission prior.  Exact authority equivalence is 100%.
- Official scorer remeasurement: task013 `memory=206, params=150, cost=356`;
  task090 `memory=916, params=133, cost=1049`;
  task134 `memory=365, params=57, cost=422`;
  task209 `memory=1832, params=253, cost=2085`; task366
  `memory=7622, params=363, cost=7985`; all are `correct=True`.

An additional task233 7308->7307 Mul/Selu identity passed the float16
exhaustive check and is raw-identical in ORT_DISABLE_ALL, but the immutable
authority itself has default-ORT runtime failures. It was therefore rejected
under the user's no-new-error campaign policy and was not staged.

The authority models already contain underspecified runtime shape metadata.
The candidates preserve those graphs except for the proven Mul/Selu swap and
one removed scalar.  No new shape witness, runtime error, or private-zero
behavior is introduced; the exhaustive bitwise proof guarantees identical
outputs for every reachable source value.

Evidence: `build.json`, `audit.json`, `exhaust_float16.json`, and
`exhaust_div_float16.json`.
- task090: two uses of `ln2`. Valid inputs have a nonempty selected maximum
  rectangle; `selected_run` is a positive integer bitset and
  `lowbit=run&-run` is a positive power of two, so both logarithms are finite
  nonnegative.
