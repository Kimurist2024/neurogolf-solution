# task333 finite/residual-support audit

## Outcome

One candidate is accepted as a non-promoted winner against immutable
`submission_base_8005.17.zip` (`c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`):
`task333_r01.onnx`, SHA `0628a573302f0a816d010482ed8b883caac7c307a27f47c9b53df85e2042a6bc`.  Actual cost is `423 -> 421` and
projected gain is `0.004739345364`.  No ZIP or protected root file was modified.

## Complete candidate inventory

The scan visited 611 task333 ONNX files and
1182 task333 members across
1259 ZIPs: 1793 source
references deduplicated to 41 SHA values.  Only four
unique SHA values are strictly lower than the actual baseline cost 423.

| SHA | cost | known right (four configs) | decision |
|---|---:|---|---|
| `63f5e12b942b…` | 412 | 0/0/0/0 | reject known |
| `9946cde435b6…` | 412 | 0/0/0/0 | reject known |
| `bb405915185d…` | 412 | 1/1/1/1 | reject known |
| `0628a573302f…` | 421 | 265/265/265/265 | accept |

All three cost-412 latent-prune files fail the mandatory known gate.  The sole
known-perfect SHA is the cost-421 sign absorption.

## Why exponential generator support can be reduced exactly

The raw generator support is not practically enumerable.  A conservative
constructive subset alone has `36*56*3^63 =
2307435527236568389690074587996832` valid
inputs.  This lane does not replace that support with fresh sampling.

The baseline has one Einsum and three input occurrences.  The removed factor is
`GE=[1,-1]`.  The candidate sets `HC_new[Z,d]=GE[Z]*HC_old[Z,d]`; the shared HC
use is compensated by `GHHT_new[t,U]=GHHT_old[t,U]*GE[U]`.  Therefore its second
use contains `GE[U]^2=1`.  Every monomial for every complete Einsum index
assignment is exactly unchanged, for every possible input tensor.  All other
initializers are byte-identical.

The complete changed-factor support is only `2*10 + 3*2*10 = 80` entries.  All
80 entries were executed and matched exactly in disabled/default ORT with
threads 1/4, with runtime errors 0 and nonfinite values 0.

## Whole-model platform and margin evidence

Known is `265/265` independently in every configuration.  In addition, two
independent generator seeds supplied 2,000 valid cases per configuration:

| configuration | truth | wrong | runtime | nonfinite | (0,0.25) | sign diff vs baseline | min positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| disable_all_threads1 | 2000/2000 | 0 | 0 | 0 | 0 | 0 | 0.99864197 |
| disable_all_threads4 | 2000/2000 | 0 | 0 | 0 | 0 | 0 | 0.99864197 |
| default_threads1 | 2000/2000 | 0 | 0 | 0 | 0 | 0 | 0.99864197 |
| default_threads4 | 2000/2000 | 0 | 0 | 0 | 0 | 0 | 0.99864197 |

Floating contraction order can change raw magnitudes; that is recorded rather
than hidden.  Across all 8,000 whole-model cases the maximum raw difference is
`0`, while sign
differences, truth errors, runtime errors, nonfinite values, and near-positive
values are all zero.

## Structural gates

Actual profiler is `0 memory + 421 params = 421`.  ONNX full check, strict
data-propagating shape inference, positive static shapes, truthful runtime
output `[1,10,30,30]`, standard domains, Conv-family UB0, lookup0, nested
graph/function/sparse0, banned-op0, and finite initializer gates all pass.  The
35-input giant Einsum is accepted only because the all-input termwise proof and
complete changed-factor residual audit above close its guarantee gap.

## Evidence

- `candidate_inventory.json`: all 1,793 source references and 41 unique SHA rows
- `strict_lower_audit.json`: actual cost and known x4 for all four lower SHA rows
- `sign_equivalence_proof.json`: exact all-input monomial proof
- `generator_support_analysis.json`: raw-support lower bound and reduction
- `factor_support_audit.json`: complete 80-entry residual x4
- `margin_support/*.json`: two-seed whole-model truth/margin x4
- `result.json`, `winner_manifest.json`: machine-readable disposition
