# task205 cost-1038 all-support proof

## Decision

**PROMOTE** exact candidate
`43c963c46bda5b444fb830b5495b4d71fb9dcf958e108954cdb9ef1064d9f9a8`
at cost **1038**. It is bitwise equivalent to the immutable cost-1042 authority
at `colq_scale` for every valid one-hot input, and the remainder of the graph is
unchanged. This is an all-support proof, not an inference from finite fresh
accuracy.

No root submission, stage, score ledger, `others/71407`, or other lane was
modified.

## Exact identities and cost

| role | SHA-256 | memory | params | cost |
|---|---|---:|---:|---:|
| authority member | `8a6acdc20a366ccbd32cf761285cbb2f1cbcf7d3d2ef8ea71d0fb5a3ed6f1468` | 1031 | 11 | 1042 |
| candidate | `43c963c46bda5b444fb830b5495b4d71fb9dcf958e108954cdb9ef1064d9f9a8` | 1027 | 11 | 1038 |

The authority archive SHA-256 is
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
The only semantic graph change is:

```text
authority: ReduceSum(row_mask) -> tall_f; Mul(tall_f, rowpow_thr) -> colq_scale
candidate: Einsum(row_mask, rowpow_thr, equation="ri,->") -> colq_scale
```

`rowpow_thr` is float32 `1.9019999504089355`, bits `0x3ff374bc`.

## Complete `row_mask` support

`row_mask` has strict static shape `[30,1]` and is produced by
`Cast(Greater(row_rough_counts, row_thr), to=FLOAT)`. Thus every element is an
exact float32 `0` or `1`, on every finite valid one-hot input. Its sum is an
integer `n` in the complete range 0..30; positions and order are unrestricted.

This is deliberately stronger than the latent generator box-height bound.
Random background affects `row_rough_counts`: in the retained 2,000-case run,
the observed mask counts were 5..11, including two count-5 cases and one
count-11 case. A proof restricted to 6..10 would therefore be unsound.

## Why ORT `Einsum` has the same float order

The ORT v1.24.1 CPU implementation in
[`einsum_typed_compute_processor.cc`](https://github.com/microsoft/onnxruntime/blob/v1.24.1/onnxruntime/core/providers/cpu/math/einsum_utils/einsum_typed_compute_processor.cc)
closes the apparent reduction-order concern:

1. `Run()` lines 1691-1738 identifies labels last seen in input zero and
   reduces those dimensions before pairwise operand processing.
2. In `ri,->`, both `r` and `i` occur only in the first input, so ORT first
   performs `ReduceSum(row_mask, axes=[r,i])` and obtains shape `[1,1]`.
3. The second input is a homogenized scalar `[1,1]`. The pairwise path at
   lines 1329-1337 and 1627-1632 therefore invokes MatMul with left/right
   shapes `[1,1,1]`; its reduction dimension is `K=1`.

Consequently the candidate does **not** add 30 separately rounded products.
Both graphs first reduce the mask, then perform one scalar multiplication.

Every partial sum of 0/1 entries is an integer in 0..30 and is exactly
representable in float32, regardless of reduction tree, thread count, or mask
order. Both reductions therefore return the same exact `n`. A `K=1` product
then has the same single correctly rounded float32 multiplication as `Mul`;
there is no multi-term accumulation. This proves identical `colq_scale` bits
for all `2^30` binary masks, including counts 0 and 30.

All downstream nodes and initializers are unchanged. Equal graph input and
equal `colq_scale` bits therefore imply equal final raw output in the same CPU
runtime configuration.

The installed audit runtime was ORT 1.24.4. The tagged-source order proof was
corroborated directly on that runtime as described below.

## Runtime corroboration

The local two-op micrographs covered every count 0..30 in three layouts
(prefix, suffix, and alternating/dispersed), 93 mask inputs total. They were run in
`ORT_DISABLE_ALL` and default optimization, each with 1 and 4 intra-op threads:

| audit | comparisons | bit/raw mismatches | errors | nonfinite |
|---|---:|---:|---:|---:|
| `colq_scale` micrograph | 93 × 4 = 372 | 0 | 0 | 0 |
| exposed full models, generator seed 24720501 | 2,000 | 0 at `row_mask`, `colq_scale`, and final raw output | 0 | 0 |

The 2,000 generator cases contained 123 distinct masks. Their count histogram
was `{5:2, 6:380, 7:415, 8:404, 9:388, 10:410, 11:1}`.

Retained evidence adds two independent known-set audits:

- The exact target SHA passes full checker, strict shape inference with data
  propagation, truthful shape trace, and all 266 known cases in disabled and
  default ORT, with zero wrong outputs or errors.
- `scripts/golf/root_reduce_scalar_fusion_scan_248/audit.json` independently
  tests the same fusion (same initializers and semantics; only topological
  placement and dummy equation labels differ) on all 266 known cases under
  disabled/default × threads 1/4. Every one of 1,064 comparisons is raw-equal
  to authority, with zero errors, nonfinite values, or shape differences.

The retained exact-target structural audit reports full checker pass, strict
data-propagating inference pass, no nonstatic inferred dimensions, no banned or
nonstandard domains, no functions/sparse/external payloads, zero Conv-bias UB
sites, and zero declared/runtime shape mismatches. Its truthful runtime memory
is 1027 and official cost is 1038.

## Fresh-run disposition

The newly requested 2 seeds × 10,000 × 4 run was stopped by parent direction
and was not represented as completed. It is not needed to bridge a support
gap: the proof above quantifies over every binary `[30,1]` mask and hence every
valid one-hot/generator input. The 2,000-case exposed-intermediate audit and
known four-configuration audits are runtime corroboration of the source-level
proof.

Final gate: **PROMOTE cost 1038**; errors 0, nonfinite 0, UB sites 0, shape
mismatches 0.

## Evidence

- `audit.json`: machine-readable identities, proof chain, exact counts, and
  final gates.
- `scripts/golf/loop_8004_42_plus20/agent_task205_private_proof_192/model_audit.json`:
  exact-target full/strict/truthful/cost/known evidence.
- `scripts/golf/root_reduce_scalar_fusion_scan_248/audit.json`: independent
  four-configuration known raw-equivalence evidence.
