# lane_rebuild_c2 — strict 7999.13 rebuild audit

## Outcome

No candidate from tasks 005, 018, 054, 133, 209, 349, or 367 satisfies all
standing gates while being strictly cheaper than the exact 7999.13 ZIP member.
The lane therefore emits an empty `winner_manifest.json`, makes no promotion,
and projects score gain `+0.0`.

The exhaustive existing-asset pass deduplicated 6,617 references into 762
models. It actual-scored 113, rejected 516 at a sound static cost floor, 118 at
the static structure gate, and 8 as unscorable. Six models were visible-correct
and cheaper, all for task018. The cheapest was cost 4733 versus 4818, but its
mandatory 3,000-case fresh result was only 2,814/3,000 (93.8%), so it is rejected.

## Per-task result

| task | 7999.13 cost | best sound / attempted cost | strict result |
|---:|---:|---:|---|
| 005 | 2325 | 2389 | sound selector is +64; reject at cost |
| 018 | 4818 | 4733 | visible 266/266, fresh 2814/3000; reject |
| 054 | 2291 | 12380 | complete rebuild is +10089; reject at cost |
| 133 | 4403 | 5416 | fresh 3000/3000, but +1013; reject at cost |
| 209 | 2218 | none | non-injective generator; no fresh-100 cheaper graph |
| 349 | 3964 | 4861 | fresh 3001/3001, but +897; reject at cost |
| 367 | 2229 | 2229 | 43 optimizer outputs are cost-equal; reject |

For task367, the exact baseline was independently rechecked at 3,000/3,000
fresh. Forty-six ONNX optimizer passes were attempted: 45 emitted files with
four unique serializations, and one lexical-lifting pass failed generation.
The two cost-equal groups covering 43 outputs retained `memory=2010`,
`params=219`, `cost=2229`, and exact visible correctness; `split_init` and
`split_predict` both fail the static single-input/single-output contract. Thus
serialization normalization does not create a score improvement.

## Gate discipline

The `neurogolf-onnx-golf` workflow determined the gate order used here:
generator source and prior failure reports first, then structural/checker and
actual official-like cost, followed only for a strictly cheaper candidate by
visible dual-path correctness, fresh generator audit, margin, and independent
structure/bias checks. This prevented spending promotion checks on sound but
more-expensive rebuilds and prevented the task018 visible-only fit from being
mistaken for a safe improvement.

The task018 candidate passed lib gold, official gold, margin 1.0, static shape
checks, and Conv-bias UB (`[]`). It fails the mandatory fresh gate by 186 cases.
The external validator's generic 500-case differential had 500 executable
cases and zero one-sided execution failures, but only 352/500 threshold-equal
outputs (148 mismatches). Arbitrary-grid equivalence cannot override the
generator-specific fresh failure. The generator itself admits ambiguity,
consistent with the prior task018 analysis and the measured 6.2% fresh failure.
The validator was invoked with `--allow-random-mismatch` only to retain these
diagnostic counts; its built-in verdict does not run the generator fresh gate
and is therefore not a promotion decision for this lane.

## Files

- `scan_results.json`: complete deduplicated pool results
- `winner_manifest_pre_fresh.json`: the one visible-only lead before fresh audit
- `winner_manifest.json`: final approved list (empty)
- `failure_manifest.json`: machine-readable rejection evidence
- `task367_optimizer_scores.json`: actual costs/static verdicts for optimizer outputs
- `task018_external_random500.json`: independent generic differential evidence

All generated models and reports are confined to
`scripts/golf/loop_7999_13/lane_rebuild_c2/`. This lane did not modify a root
submission, score file, handcrafted model, or promotion artifact.
