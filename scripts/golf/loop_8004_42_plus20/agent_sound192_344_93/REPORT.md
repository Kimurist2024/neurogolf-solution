# SOUND task192/task344 lane — authority 8006.61

## Accepted isolated probe: task192

The exact polynomial task192 model reduces the competition-profiler cost from
`1609` to `1307` (`memory=168`, `params=1139`). The conservative projected
score gain is `ln(1609/1307) = +0.20787843336816125`.

Candidate:
`candidates/task192_exact_poly.onnx`, SHA-256
`c3cbaf44d962ca72e15514da1b32c121ee489d153ef39d38b7101f09576e92b6`.

Verification is fail-closed and complete:

- immutable authority ZIP SHA-256 `9085e2f7...fa4118` and authority task192
  member SHA-256 `e7f9a11b...ede10c` checked before comparison;
- ONNX full checker, strict shape inference/data propagation, all static
  positive shapes, standard domains, banned-op/nested/function/sparse checks,
  and Conv-bias UB scan all pass;
- the final mathematical `Einsum` has 12 inputs, below the giant-input gate;
- known corpus: `265/265` in each of four runtime configurations
  (optimization disabled/default, threads 1/4), with zero runtime errors,
  zero nonfinite values, and minimum positive margin `1.0`;
- fresh generator: `5000/5000` at seed `192800661` and `5000/5000` at seed
  `192930007`; the independent readable rule is also `5000/5000` for both;
- direct runtime shape trace covers all five node outputs with zero declared /
  actual mismatches and zero nonfinite values;
- exhaustive local-count sign proof covers all 163 possible count tuples with
  zero failures.

The exact construction selects the most frequent nonzero color `A` with the
lowest-color tie break. At each cell it computes
`P = nonzero(center) * horizontal_count(A) * vertical_count(A)`. Thus `P>0`
is exactly the rule predicate. Background receives `B-9P`, where `B` is the
product of the horizontal and vertical in-grid window sizes. Since
`1 <= B <= 9`, background is positive exactly when `P=0`. The selected-color
channel receives `P`. No examples, coordinates, or expected outputs are
stored in the model.

The first contraction ordering was mathematically correct but slow. Reordering
the identical equation to contract each color/direction first reduced runtime
from about five seconds to about three milliseconds per visible example
without changing cost or semantics.

Evidence: `audit/task192_exact_poly.json`; concise adoption data:
`probe_manifest.json`.

## task344 status

The authority task344 member is the verified cost-197 rank-4 local-rule model
(SHA-256 `d0902dc6...f08019f`). No cheaper SOUND replacement is accepted.

The new cost-188 probe removed the full-rank `S[3,3]` factor and distilled the
remaining `H/V/M` factors from two algebraic starts. After two restarts and 24
epochs its best validation/visible-guard cell mismatch counts were 7,481 and
2,939. The serialized ONNX probe then scored `0/266` known examples, so it was
rejected before fresh testing. Historical cost-181 shared-V, color-rank-3, and
spatial-rank-3 cost-191 probes were also inexact and remain excluded.

All authority factors are at their structural rank limits (`H` rank3, `V`
rank4, `B` rank4, `S` rank3, `M` rank4). The 20 zero tail columns of `B` cannot
be deleted while preserving the declared 30x30 direct output; constructing
them dynamically costs more memory than it saves. There is therefore no
successful strict coefficient-sharing, factor-rank, dead/alias, no-op, or
output-route shave in this pass. Evidence:
`audit/task344_no_s_rejected.json`.

This lane leaves task344 and every protected/root artifact unchanged.

No root submission ZIP, score ledger, CSV, `artifacts/`, `handcrafted/`, or
`others/` file was written.
