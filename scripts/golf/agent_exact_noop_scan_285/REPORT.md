# agent_exact_noop_scan_285

## Decision

`NO_EXACT_NOOP_WINNER`

The immutable `submission_base_8009.46.zip` authority was scanned across all 400
tasks. No candidate satisfied the complete admission chain, so no ONNX candidate
was retained and no known/fresh comparison was eligible to run.

Authority SHA-256:
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.

## Scope and ordering

- The assigned active22 snapshot was excluded exactly as requested.
- During the run, the shared manifest had concurrently gained task355 (23 active
  tasks total). Task355 was also excluded as a race-safety measure; neither the
  manifest nor any root artifact was modified.
- The private/unsound-route catalog was excluded. Lookup, giant-initializer,
  giant-Einsum, nonstandard/custom-domain, non-static-shape, nonfinite-initializer,
  and Conv-bias-UB checks were enforced by the structure audit.
- Authority costs 150--500 were ordered first: 112 tasks total, comprising 39
  structurally eligible scans, 51 authority-structure exclusions, 14 private-route
  exclusions, 7 assigned-active exclusions, and 1 concurrent-active exclusion.
- Across all costs: 156 tasks were structurally eligible for transformations, 173
  authorities failed the conservative structure/route gate, 48 were private-route
  exclusions, 22 were assigned-active exclusions, and task355 was the one
  concurrent-active exclusion.

## Transformations searched

The scanner systematically attempted dead node/initializer cleanup, byte-identical
initializer sharing, Identity removal, same-dtype Cast removal, Add/Sub zero,
Mul/Div one, zero Pad, identity Transpose/Reshape/Expand/Squeeze/Unsqueeze, adjacent
inverse Transpose, and adjacent Squeeze/Unsqueeze cancellation. Combined cleanup was
also applied to a fixed point. Canonical graph input/output names were preserved.

There were 24 profile/action hits collapsing to 8 unique serialized candidates:

| Task | Profile | Authority -> actual | Result |
|---:|---|---:|---|
| 34 | neutral Mul-by-one | 511 -> 510 | runtime trace failed (ScatterElements rank error) |
| 39 | dead node | 42 -> 41 | runtime trace exposed declared/actual shape mismatches |
| 75 | Identity | 328 -> n/a | checker/strict data propagation inference failure |
| 111 | dead node | 89 -> 88 | runtime trace exposed shape mismatches and 12 undeclared intermediates |
| 214 | Identity | 85 -> n/a | checker/strict data propagation inference failure |
| 264 | duplicate initializer | 344 -> 343 | runtime trace exposed extensive declared/actual shape mismatches |
| 269 | Identity | 31 -> n/a | checker/strict data propagation inference failure |
| 289 | Identity | 32 -> n/a | checker/strict data propagation inference failure |

Thus, 4 candidates failed full checker/strict shape before actual-cost admission,
and the 4 strict-lower actual candidates all failed the mandatory runtime
intermediate shape-cloak gate. Eligible candidates after these gates: **0**.

## Conditional equivalence testing

The configured next stage compares candidate and authority raw output fingerprints,
sign outputs, truth, runtime/nonfinite/shape/small-positive counts, and configuration
stability on known cases under disabled/default optimizations at threads 1 and 4.
Only a complete known pass proceeds to two fresh generator seeds with 2,000 cases
per seed under all four configurations and the `truth >= 90%` rule.

Because no candidate passed the prerequisite structure, actual-cost, and truthful
runtime-trace gates, known runs executed: **0**; fresh runs executed: **0**. This is
a conditional non-execution, not a claimed equivalence result.

## Artifacts and protection

- `scan.py`: reproducible scan and conditional verification implementation.
- `evidence.json`: all-400 inventory, exclusion evidence, transformation actions,
  structure audits, actual costs, and runtime-trace rejection details.
- `candidates/`: empty, because there is no complete pass.

No automatic promotion was used. `submission.zip`, `all_scores.csv`, repository-root
artifacts, and `others/71407` were not written by this lane.
