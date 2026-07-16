# task343 SOUND cost-172 redesign (base 8005.17)

## Result

No admissible task343 replacement was found.  The current ZIP member remains
unchanged and **must not be replaced** by either locally cheaper model.

- Authority ZIP: `submission_base_8005.17.zip`
- Authority ZIP SHA-256: `c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`
- Current `task343.onnx` SHA-256: `7d64c3eda1167f322d8981531e433e7195e54d48e16e29c771b52a379af17ab1`
- Current cost: `173` (`memory=140`, `params=33`)
- Required exact replacement: `<=172`
- Safe candidate at `<=172`: none
- ZIP/CSV/score ledgers changed: no

This is a no-adoption result.  It prevents the known cost-172 approximation
from being promoted merely because it passes visible gold.

## Authoritative rule

The rule was recovered from `inputs/sakana-gcg-2025/raw/task343.py` and checked
against `inputs/arc-gen-repo/tasks/task_d8c310e9.py` plus `common.py`.
For every row `r`:

```text
L = 8 if r[0:4] equals r[4:8] or r[8:12], otherwise 6
output_row = (r[0:L] repeated three times)[0:15]
```

The generator draws a block of width 3 or 4.  Alternating reversal gives full
period 6 or 8.  Thus period 6 is exact for the width-3 family and period 8 is
exact for the width-4 family; examples for which both periods are equivalent
are intentionally treated as don't-care classifier cases.

No lookup, giant-input construction, archive cloak, custom domain, nested
graph, or private-zero-only acceptance was used.

## Candidate comparison

| file | cost | known gold | fresh evidence | verdict |
|---|---:|---:|---:|---|
| current/base173 | 173 | pass | 4970/5000 | retain only as authority; not a new safe candidate |
| handcrafted172 | 172 | pass | 4976/5000 | reject |
| exact178 control | 178 | pass | 5000/5000 | sound but cost regression; reject |

Files and hashes:

- `scripts/golf/scratch_codex_7994/task343_sound/base173.onnx`:
  `7d64c3eda1167f322d8981531e433e7195e54d48e16e29c771b52a379af17ab1`
- `artifacts/handcrafted/task343.onnx`:
  `c1047d40b875d37a7a9e28a52a47e2c569f5156924691118082aaca4ed5198e6`
- `scripts/golf/scratch_codex_7994/task343_sound/archive_89ba_4conv.onnx`:
  `b47938285ea00b04aebea8709dd448c9983f0e3c8c6284050314097af0525c1b`

Official-like local scoring reconfirmed costs `173`, `172`, and `178`
respectively.  All three pass repository known gold, demonstrating why visible
gold alone is not a sufficient adoption gate for this task.

## Exact control and cost floor

The exact-178 control uses four scalar dynamic correlations:

```text
q4  = shift-4 self-correlation
q11 = shift-11 self-correlation
z7  = visibility-width <= 7 bit
z11 = visibility-width <= 11 bit
use period 6 iff q11 + 2*z7 + 35*z11 > q4
```

Its graph is `Conv x4, Sum, Greater, Squeeze, Where, Mod, Gather`, with no Conv
bias and only finite initializers.  The coefficient is represented by repeated
inputs to `Sum`, not learned data.

The existing output route consumes a fixed cost of 157 before the decision:

- parameters: coordinate indices 30 plus periods 6 and 8 = 32;
- memory: index vector 120, selected period 4, squeezed condition 1 = 125.

Therefore cost 172 leaves at most 15 decision bytes.  A normal exact candidate
must fit one of the compact families below; four independent float32 probes
already consume 16 bytes before comparison.

## New searches in sound77

The previous SOUND work searched row-aligned correlations.  This run expanded
the actual low-cost design space as follows:

1. `search_expanded.py`: 930 unique row/column/visibility-point/bottom-point
   features on 1,200 generated examples.  Exhausted single relations and
   three-feature two-comparison `AND`/`OR` formulas.  Result: none.
2. `search_full_thresholds.py`: exhaustively generated all 53 distinct
   vertical affine maps times all 1,058 distinct horizontal affine maps on the
   real 5x15 support, or 56,074 scalar dynamic-Conv features.  Compared every
   feature with reusable truthful constant-Conv probes 1, 5, 15, and 75 and
   screened shared-constant `AND`/`OR` formulas.  Result: none.
3. `search_full_anchors.py`: screened all 56,074 features against 129 shared
   anchors comprising exact-control probes, visibility probes, simple shifts,
   and correlation-ranked features.  Result: none.
4. `search_cast_relation.py`: accounted for the lower-memory construction
   `Cast(nonzero Conv) AND/OR/XOR (Conv relation Conv)`, which also fits exactly
   in the 15-byte allowance.  Screened 302 anchors against all 56,074 features
   using the proven z11 boundary bit.  Result: none.
5. `search_cast_relation_masks.py`: broadened the Cast search to 300 distinct
   universal Cast masks and 600 relation anchors.  Result: none.
6. Full truthiness census over the 56,074 features found 11,978 distinct
   nonzero signatures, 1,935 universal-period-6 signatures, only 3 signatures
   that reject all period-8 cases, and no exact one/two-Conv `AND`, `OR`, or
   `XOR` classifier.

Discovery samples contained only6/only8 hard labels and ignored the small set
where both periods are correct.  Any sampled hit would still have required
independent fresh seeds, four ORT configurations, strict shape inference,
official cost, and a true-rule proof.  Since no sampled exact hit exists, no
ONNX candidate advanced to those adoption gates.

## Adoption decision

`winner = null`.  Keep the authority ZIP member byte-for-byte.  The cost-172
visible-gold model has known generator counterexamples, while the cost-178
model is safe but worsens score.  Neither is eligible under the requested
no-error policy.
