# lane_c3 — strict 7999.13 audit

## Outcome

No strict winner was found for tasks 036, 044, 090, 125, 205, 319, or 361.
The final winner manifest is empty, projected gain is `+0.0`, and no root
submission, score file, handcrafted model, or promotion artifact was changed.

The harvest reduced 6,579 candidate references to 758 unique models. Sixty-six
models were actual-scored; 619 were rejected at a sound static cost floor, 63
at structural validation, and 3 were unscorable. Every one of the 46 available
single ONNX optimizer passes was also tried independently against all seven
exact baselines; none produced a strictly cheaper visible-correct graph.

## Strict results

| task | base cost | base fresh | best sound / cheaper probe | decision |
|---:|---:|---:|---:|---|
| 036 | 325 | 2986/3000 | sound 446, 3000/3000 | cost reject |
| 044 | 1087 | 2938/3000 | non-injective generator | rule reject |
| 090 | 1050 | 2946/3000 | cheaper 986, 2812/3000 | fresh reject |
| 125 | 1050 | 2999/3000 | sound 1167, 3000/3000 | cost reject |
| 205 | 1042 | 2949/3000 | cheaper 1038, 2949/3000 | fresh reject |
| 319 | 1023 | 2921/3000 | non-injective generator | rule reject |
| 361 | 968 | 3000/3000 | manual shave 1363 | cost reject |

Task090 had five distinct cheaper visible-correct models at costs 986, 1004,
1016, 1029, and 1036. Their fresh failure counts were respectively 188, 122,
116, 83, and 82 out of 3,000. Task205's cost-1038 candidate retained exactly
the base failure count, 51/3,000. These are visible-corpus optimizations, not
healthy rule implementations.

The generator-derived task036 cost-446 and task125 cost-1167 models were
independently rerun and both passed lib gold, official gold, margin, and
3,000/3,000 fresh. They establish safe local alternatives, but neither is
strictly cheaper than the exact 7999.13 payload.

## Manual task361 shaves

Task361 was the only baseline to pass 3,000/3,000 fresh, so manual work focused
there after the full harvest.

1. Folding the input-independent `[10,10]` `Concat` into an initializer would
   remove a 16-byte intermediate, but constant propagation exposed the true
   `CenterCropPad` shape and strict checker rejected the graph. It was not
   retained as a candidate.
2. Replacing `CastLike(idx_src, idx_ref)` with `Cast(to=INT32)` removes a
   type-only initializer and remains 3,000/3,000 correct. Official profiling,
   however, charges the exposed runtime index tensor and raises cost from 968
   to 1363, so it is rejected.

No dead-node deletion, runtime-exception graph, or unverified cloak was accepted.

## Gate discipline

The `neurogolf-onnx-golf` process controlled the audit order: compile the
generator rule, enforce checker/strict-static/banned-op/bias gates, measure
official-like runtime cost, require known exactness and domain fresh 3,000/3,000,
then run independent arbitrary-grid raw/threshold differential only for a
strictly cheaper survivor. No graph reached both the cost and fresh gates, so
random differential was correctly not used to override a terminal failure.

Evidence is in `scan_results.json`, `optimizer_sweep.json`,
`failure_manifest.json`, and the empty `winner_manifest.json`.
