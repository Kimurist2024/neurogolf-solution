# 8023.08 high-cost / unexplored lane

- Immutable authority: `submission_base_8023.08.zip`
- Authority SHA256:
  `0e29e8d57f7ac58136a9574351c9c6f3056f9debf6eeee9c181c8f2e9fac690a`
- Scope: every ledger task at cost >=300, excluding authority-integrated
  tasks 132/168/226/345 and separately staged task275.
- New source census: every ONNX drop in `others/71609`.
- Additional census: current-authority unary/exact simplifiers plus the broad
  mechanical rewrite suite (unused outputs, initializer aliases, strict no-ops,
  Einsum precontraction/fusion and safe ONNX optimizer passes).
- Required gate: local and official gold exact, strict static graph, stable
  margin, fresh 2,000 x 2 at 100%, and all runtime/nonfinite/shape/
  small-positive counters equal to zero.

## New winner

| task | authority -> candidate | gain | known | fresh | raw identity |
|---:|---:|---:|---:|---:|---|
| 200 | 346 -> 342 | +0.011628 | exact | 2000/2000 x2 | yes |

Candidate:
`focus/candidates/task200_POLICY90_cost342_c659ae401e4c.onnx`

Candidate SHA256:
`c659ae401e4c92a53cad5d4a251aac3ad562c7e919dcd5b4b82c42ed63c8a07d`

The rewrite removes `Cast(seed_f -> int64)` and sends `seed_f` directly to
`OneHot`.  The task generator fixes the grid to size 10 and places exactly one
colored cell at integer column 0..9.  The graph computes
`seed_f = seed_prod / color_f = column`; all operands are exact small integers,
so the quotient is an exactly representable integer.  The candidate is also
raw-bit-identical to the authority on all known cases and passed both fresh
streams with minimum positive margin 1.

## Probe

- Path: `submission_PROBE_task200_cost342.zip`
- SHA256:
  `5df9b59490829c07c190f5956df5ba5af961cefaf919678b9201d51102ac504a`
- Changed member: only `task200.onnx`
- Projection: +0.011628, giving 8023.091628 from 8023.08

See `probe_manifest.json` and `focus/task200.json` for machine-readable
evidence.

## Exhaustion result

The three-way high-cost screen found no other known-exact strictly-cheaper
candidate.  The separate broad exact-rewrite scan also produced zero finalists.
No known-private-zero candidate was promoted.

## Root protection

This lane did not update `submission.zip`, `all_scores.csv`, or
`best_score.json`. Their SHA256 guards were identical before and after the probe
build.
