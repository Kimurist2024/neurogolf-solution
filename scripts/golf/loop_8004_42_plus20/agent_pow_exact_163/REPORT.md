# Pow exact scan 163

## Outcome

No eligible rewrite exists in the immutable LB 8009.46 authority. The safe
winner set is empty and projected gain is `+0.0`. Root `submission.zip`,
`all_scores.csv`, `others/`, and docs were not modified.

Authority SHA-256:
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
All 400 expected task members were present and independently hashed.

## Exhaustive scan

All 400 models and every graph node were scanned. Eleven tasks contain 18
`Pow` nodes: 002, 148, 153, 161, 182, 201, 208, 250, 259, 319, and 328.

The required candidate definition was exact and narrow: the second `Pow` input
must be an initializer tensor containing exactly one element, and that element
must equal `2`, `0.5`, or `1`. The result is empty:

| exponent classification | nodes | disposition |
|---|---:|---|
| initializer, but more than one element | 7 | ineligible |
| exponent is not an initializer | 9 | ineligible |
| scalar initializer, but value is not 2/0.5/1 | 2 | ineligible |
| eligible scalar initializer 2/0.5/1 | **0** | no candidate |

The seven non-scalar initializer cases are vector/matrix broadcasts such as
`[0,1]` and `[0,0.25]`; replacing the whole node with `Mul`, `Sqrt`, or
`Identity` would not be equivalent. The only scalar-initializer `Pow` nodes are
the two task250 nodes, both with exponent approximately `0.4125`, so none of
the requested identities applies. Nine other nodes compute the exponent at
runtime.

## Validation disposition

Candidate generation produced zero files. Consequently there is no model on
which checker/full, strict inference, competition actual cost, four runtime
modes (`default`, `disabled`, `extended`, `minimal`), known raw/sign
equivalence, runtime/nonfinite/margin, private-zero, or fresh gates can be
meaningfully run. These stages are recorded as
`vacuous_no_candidates`, not as a validation pass.

The scan did not broaden the request to Constant-node exponents, dynamic
exponents, or decomposition of vector-valued exponent tensors. Doing so would
create a different transformation class and would require a new soundness
audit.

## Evidence

- `scan_pow.py`: reproducible fixed-SHA all-400 scan and exact candidate builder.
- `result.json`: all 18 `Pow` rows with task, node, inputs, exponent source,
  initializer dtype/shape/value where applicable, and rejection reason.
- `winner_manifest.json`: empty safe promotion manifest.

