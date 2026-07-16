# 8019.75 low/mid strict lane

## Outcome

The strict winner remains task175 at cost 131.  It is rebased directly on the
fixed `submission_base_8019.75.zip` authority without modifying any root score
or submission file.

| task | authority cost | candidate cost | projected gain | disposition |
|---:|---:|---:|---:|---|
| 175 | 140 | **131** | **+0.066445099408** | **STRICT** |
| 328 | 427 | 352 | +0.193152837631 | private-zero probe only; strict-margin reject |

The strict task175 projection is `8019.816445099408` before combining with
other lanes.

## Strict task175

Candidate:

`scripts/golf/restart8018_91_lane_low/candidates/task175_gauge_s_factor_reuse.onnx`

SHA-256:

`22fe38f6428dbc2f98b7135825325044f1898a7da23e2bea9b7584d97bfe4265`

The candidate is the previously completed exact tensor-network rewrite:

1. absorb the live W/V gauge into TA/TB;
2. replace the shared selector S by the exact `S = T @ Msel` factor reuse.

Its pinned evidence remains:

- full checker and strict data-propagating shape inference pass;
- canonical input/output `[1,10,30,30]`;
- local and official gold 266/266 exact;
- cost `memory 0 + params 131 = 131`;
- minimum positive raw margin 0.25, false maximum 0;
- independent seeds 801891177 and 801891178, 2000/2000 each under
  ORT-disable threads 1 and 4;
- errors, nonfinite values, shape mismatches and small positives all zero.

The 8019.75 member for task175 is byte-identical to the pinned cost-140 source,
so the rewrite rebases without a model drift assumption.

Rebased strict probe ZIP:

`submission_STRICT_task175_cost131.zip`

SHA-256:

`ee69a3d4e93512a719c42a40858e79a4b24a4e6bd568b3fd3af93a9556f6fe89`

It has 400 unique members and changes only `task175.onnx`.

## task328 exhaustive private-zero result

The current LB-white cost-427 model has a historical cost-352 relative.  The
cost-352 graph passes full checker, strict static inference, complete known
gold and fresh-100 sign checks, but its raw margin is near zero.  A one-use
terminal `Rflip` factor permits mathematically uniform power-of-two scaling.

The `2^40` candidate was exhaustively audited over all generator states.  The
71,136 states reduce exactly to 143 geometric representatives because the
graph is equivariant under every permutation of nonzero colors.  Both
ORT-disable threads-1 and threads-4 configurations returned:

- sign-correct 143/143;
- wrong/runtime/nonfinite/false-positive counts zero;
- maximum false raw value 0;
- maximum absolute raw value `2.380882120981465e29`;
- minimum positive raw value only `1.8677450372594196e-17`;
- 964 positive cells in `(0,0.25)`.

Thus normal strict admission is impossible by output scaling.  Reaching 0.25
requires about another `2^54`, while the observed maximum would exceed
`4e45`, above finite float32.  The candidate is quarantined and must not be
merged as a strict winner.

For the user's explicit full-support private-zero exception only, the isolated
probe is:

`submission_PROBE_PRIVATEZERO_task328_cost352.zip`

SHA-256:

`304a84d5754a14e8309cdc2e309751eaadd045d8a69fe7668a978507a550788b`

The combined task175/task328 probe is
`submission_PROBE_task175_131_task328_352.zip`, SHA-256
`f790d26b5da112a107ff0497533bce9cc100910100bc23b62be7ff1040fa47b3`.
It is probe-only because task328 does not meet the strict margin gate.

## Other newly-white tasks

- task102: the cost-491/current family is not generator-exact (fresh samples
  fail); the truthful reconstruction costs 4069, so no strict lower candidate.
- task205: the cost-778 current model is not generator-exact; known truthful
  constructions remain more expensive, so no strict lower candidate.

## Evidence and scope

- `task175_cost131_evidence.json` in the prior 8018.91 lane: strict task175
  audit;
- `task328_scale2p40_orbit_audit.json`: exhaustive task328 support audit;
- `task328_r01_scales_build.json`: fixed-SHA power-of-two constructions;
- `probe_manifest.json`: archive membership, changed-member and SHA checks.

This lane wrote only under `scripts/golf/restart8019_75_lane_low/`.  Root
`submission.zip`, `all_scores.csv`, `best_score.json` and score pointers were
not modified.
