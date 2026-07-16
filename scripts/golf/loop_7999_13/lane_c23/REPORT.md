# Lane C23 report: task237 / task378

## Decision

No model was promoted. The lane's actual score gain is **0.00**.

The promotion gate required a candidate to be strictly cheaper than the exact
7999.13 baseline, pass all known examples under both ORT modes with zero
errors, and remain a dense/static truthful true-rule compiler before fresh
5,000 and external-validator testing. No candidate reached the first gate.

## Baselines

| task | cost | memory | params | SHA-256 | known ORT disable-all | known ORT default |
|---:|---:|---:|---:|---|---:|---:|
| 237 | 529 | 413 | 116 | `d8ef7011cb912b1752d471535b55110c0d3474d517388d73e92e950abda58ff4` | 266/266, 0 errors | 266/266, 0 errors |
| 378 | 525 | 468 | 57 | `3e66557d91ea20496174cb741299b4b0cf6944c74dd714fd71f0a6fa032fc416` | 267/267, 0 errors | 267/267, 0 errors |

The task237 baseline is fully auditable: ONNX full checker and strict shape
inference pass, the runtime tensor tracer reproduces the official 413-byte
memory cost with no declared/actual mismatch, and the model contains no custom
domain, local function, sparse initializer, banned operator, lookup signature,
giant Einsum, or unsafe convolution bias.

The task378 baseline passes the ordinary known-example checks, but it is not a
truthful sound control. When every intermediate is exposed, ORT reports a
buffer-reuse shape conflict (`{1,10,1,1}` versus `{1,10,30,30}` at `Mul`). It
also relies on three `CenterCropPad` shape declarations. This is why no result
from that lineage was accepted merely because ordinary ORT happened to run it.

## True-rule references

The readable references were rebuilt from the ARC generator rules, not fitted
to examples.

| task | visible | fresh generator | extra bounded check |
|---:|---:|---:|---:|
| 237 | 266/266 | 5,000/5,000 | - |
| 378 | 267/267 | 5,000/5,000 | 13,712/13,712 geometry/color-invariant cases |

All rows had zero wrong answers and zero errors. Exact commands and seed policy
are recorded in `reference_audit.json`.

## Candidate results

### task237

Three independent dense/static true-rule rebuilds cost 542, 543, and 544.
All three pass both ORT modes on 266/266 known examples with zero errors, full
checking, strict inference, and exact runtime memory tracing. The best rebuild
is therefore 13 cost units worse than the 529 baseline.

The current audit also rechecked the prior finite-state, separability,
QLinearConv decoding, width-flag, kernel factorization, duplicate-initializer,
and expression-dedup searches. None produced an actual cost below 529. This is
an empirical search result, not a mathematical lower-bound claim.

### task378

The eight archived compact lineages have actual costs 543, 554, 546, 548, 529,
531, 594, and 614, so none is below 525. More importantly, the first six fail
truthful all-intermediate tracing through buffer-reuse or duplicate-node-name
errors; the last two expose 12 runtime/declaration mismatches.

The sound dense/static controls are much larger:

| control | cost | memory | params | SHA-256 |
|---|---:|---:|---:|---|
| bounded K12 ray scatter | 1651 | 1540 | 111 | `2fa9656d07b3be88e8b769c79f1bc2ee7652a54b784c97a8a8f4fee89c00acd0` |
| full-mask construction | 1701 | 1605 | 96 | `0c7ae923daf0ed41dfe3952edbaf0a2bce825ed4603af77f4635eb08649f47dc` |

Both controls pass 267/267 known examples under both ORT modes, full checking,
strict inference, exact runtime tracing, and the static safety scan. The K12
control is 1,126 units worse than the 525 baseline, so promotion is impossible
under the strict-cheaper rule.

## Validation gate outcome

There was no strictly cheaper candidate. Consequently, no candidate fresh
5,000 run and no external-validator run were performed. The reference fresh
5,000 runs above verify the decoded task rules; they are not being presented as
candidate acceptance tests.

Complete per-model measurements, ORT totals, tracer results, operator scans,
and hashes are in `candidate_audit.json`. The machine-readable decision is in
`decision.json`.

## Root integrity

The forbidden root artifacts remained unchanged:

- `submission_base_7999.13.zip`: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- `all_scores.csv`: `3f9533f472a2153e12daeea4936aa7be3f47902a8fdb1621c31f778f6d009665`
- `best_score.json`: `551409d40c18ef80a9ae7e89a6a0e567aa2474924018225e29639b32c0627e72`
- `artifacts/handcrafted` aggregate (402 files): `5344ea88ff3e24509ed49fbc51b613ced484c8000513ee45060a6ce0b7ddbf69`

All C23 files are confined to `scripts/golf/loop_7999_13/lane_c23/`.
