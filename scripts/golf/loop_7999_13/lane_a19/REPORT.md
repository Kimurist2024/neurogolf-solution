# A19 sound audit — task118 and task173

## Outcome

- Exact source: `submission_base_7999.13.zip`
- Exact SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Baseline costs: task118 = 3914, task173 = 3525
- Retained byte-distinct history audited: 16 models
- Safe winners: 0
- Projected gain: `+0.0`

No root ZIP, CSV, score ledger, handcrafted model, or shared submission was
modified. Fresh 5000 was not started for a new candidate because none passed
the mandatory pre-fresh gates.

## task118: 100% is information-theoretically impossible

The true generator draws one to four radius-2/radius-3 plus crosses over random
gray static. Cross cells that were black become red; cells that were already
gray become cyan and are then hidden back to gray in the input. The output
restores those hidden cyan cells.

This input-to-output relation is not a deterministic function. If every cell
of a generated cross was already gray, the input contains no red witness and
is identical to ordinary gray static, while the output still depends on the
hidden cross. Radius 2 and radius 3 can also become observationally identical
when the radius-3 endpoints are gray. The earlier generator-derived A2 audit
found 148 redless-cross grids in 30,000 draws and recorded:

| Model/reference | Fresh result |
|---|---:|
| Exact cost-3914 baseline | 4393/5000 |
| Best readable observable rule | 4852/5000 |
| Cost-3911 historical r01 | 4288/5000 |

Therefore no deterministic ONNX can meet the assigned 5000/5000 requirement.

The current A19 history audit independently rejects every cheaper artifact.
r01 has actual cost 3911 but static floor 491, thirty declared/runtime shape
mismatches, and default ORT fails all 267 known executions. r02-r07 are
known-wrong or buffer-shape runtime failures and remain shape-cloaked. r08 is
known-complete but costs 3915, above the exact baseline, and is also cloaked.

## task173: cheap history is cloak/error history

The true rule dynamically learns, from one full exemplar per family, each
family's outer color, center color, and X/plus/horizontal/vertical geometry.
It then completes center-only and outer-only copies. This requires dynamic
family routing; it is not a fixed output lookup.

The prior generator-compiled standard control is known 266/266 and fresh
5000/5000 with zero errors, but costs 53570. It is the relevant truth control,
not a cheaper replacement for the exact cost-3525 member.

The apparent cost-2448 r01 is terminally unsafe:

- inventory static cost 1458 and actual cost 2448 disagree;
- output is declared `[1,1,1,1]` although runtime output is
  `[1,10,30,30]`;
- an all-intermediate runtime trace fails on a declared 28x28 versus actual
  30x30 buffer reuse;
- the Conv-bias gate sees QLinearConv bias length 10 against the cloaked
  inferred output channel 1.

It passes known examples in ordinary sessions, but that cannot override the
explicit shape/value and actual/static violations.

Variants r02, r03, r04, and r06 cannot create the required ORT session because
their TopK kernel resolves to the unsupported TopK(11) implementation. r05 is
known-complete and actual cost 3513, but its inventory static cost is 2217, its
output is likewise scalar-cloaked, and the runtime shape trace is invalid.
r07 and r08 are known-complete but actual costs 3706 and 4116 are not cheaper.

## Gate disposition

All sixteen historical models were checked for full ONNX validity, strict
shape inference with data propagation, complete known behavior under both ORT
modes, actual cost, static cost, standard domains, giant Einsum, lookup-like
initializers, Conv-family bias consistency, runtime shape/value contradictions,
and unsupported TopK sessions. `history_audit.json` has `pending: []`.

Since no candidate survived those gates, running 5000 cases cannot create an
eligible winner. `winner_manifest.json` therefore keeps `accepted: []`.
