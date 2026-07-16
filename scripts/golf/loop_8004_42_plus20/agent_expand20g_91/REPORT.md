# expand20g_91 — 20-task repository-wide history scan

## Outcome

The target set was expanded to 20 tasks and every repository ZIP member plus
every loose ONNX was SHA-deduplicated.  The exact LB8006.61 champion is the
read-only `submission.zip` with SHA256
`9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`.
The lane's 20 extracted baselines are byte-identical to that authority and to
the read-only `others/71403/lb_verified_8006.61/submission.zip` mirror.

There is no locally fixed winner.  One new task066 payload remains
`LB_PROBE_REQUIRED`; it was not merged.  Three other task066 payloads are
byte-exact matches to networks already proven LB-black and are not new probes.

## Search coverage

- Targets: `066,046,117,270,165,310,238,156,035,069,354,019,237,378,368,284,363,034,089,125`
- ZIP files scanned: 1,259
- Target members observed inside ZIPs: 23,674
- Loose ONNX files enumerated: 285,747
- Target loose ONNX observations: 12,213
- Unique non-authority target SHA payloads: 1,034
- Conservative could-be-cheaper payloads sent to isolated actual audit: 99
- Actual-audit decisions: 83 not strictly lower/unscorable, 11 known/runtime
  reject, 1 isolated process crash, 3 exact known LB-black, 1 LB probe
- Exclusions from discovery: the lane itself, `.git`, `.venv`, `node_modules`,
  and protected `others/71403`

| task | authority cost | unique SHA | actual-audit leads | result |
|---:|---:|---:|---:|---|
| 066 | 677 | 63 | 4 | 3 exact LB-black, 1 LB probe |
| 046 | 631 | 65 | 0 | no strict-lower lead |
| 117 | 606 | 88 | 14 | no actual improvement |
| 270 | 594 | 50 | 0 | no strict-lower lead |
| 165 | 592 | 93 | 9 | 2 known/runtime reject, 7 no improvement |
| 310 | 566 | 37 | 1 | no actual improvement |
| 238 | 559 | 44 | 2 | no actual improvement |
| 156 | 556 | 34 | 0 | no strict-lower lead |
| 035 | 545 | 32 | 1 | known/runtime reject |
| 069 | 259 | 50 | 2 | no actual improvement |
| 354 | 537 | 48 | 3 | no actual improvement |
| 019 | 536 | 41 | 1 | isolated process crash |
| 237 | 529 | 38 | 3 | known/runtime reject |
| 378 | 525 | 62 | 8 | no actual improvement |
| 368 | 521 | 38 | 0 | no strict-lower lead |
| 284 | 518 | 45 | 15 | no actual improvement |
| 363 | 513 | 60 | 0 | no strict-lower lead |
| 034 | 511 | 34 | 0 | no strict-lower lead |
| 089 | 1349 | 62 | 6 | 4 known/runtime reject, 2 no improvement |
| 125 | 1050 | 50 | 30 | 1 known/runtime reject, 29 no improvement |

The `strong_exact_class` discovery flag found three task238 and two task069
payloads in the broader inventory, but none became an actual strict-lower fixed
winner under the truthful profiler audit.

## Remaining task066 LB probe

- SHA256: `3a31ce1c686644c68b4f9177f88bf74b7fea4756c0cd1d995850741f702fe050`
- Path: `candidates/history_prefilter/task066_h0004_3a31ce1c6866.onnx`
- Actual cost: 583 versus authority 677
- Projected gain if LB-white: `ln(677/583) = +0.1494840866`
- Known suite: 266/266 in each of DISABLE_ALL/default × threads 1/4
- Runtime errors/nonfinite/near-positive values: 0; minimum positive output 1.0
- Structure: full checker, strict data propagation, canonical IO, standard
  domain, truthful runtime shapes in both ORT modes, no lookup/cloak op, no
  banned/nested/function/sparse content, Conv-family UB count 0
- Risk: seven Einsum nodes with maximum 61 inputs; not computationally exact to
  the authority
- Independent fresh seed 91066091: 481/500 = 96.2% in both ORT modes
- Independent fresh seed 91066092: 479/500 = 95.8% in both ORT modes

This clears the user-authorized 90% local threshold, but 40/1,000 fresh cases
still fail.  Local fresh is not an LB/private guarantee, so the payload remains
probe-only and is not classified fixed or white.

## Exact known-LB-black payloads

| cost | SHA prefix | fresh minimum | exact history |
|---:|---|---:|---|
| 368 | `d909159d1643` | 0/500 | others/1200 task066; LB bisect/accounting black |
| 582 | `65a1b5888e49` | 0/500 | others/1102 task066; single-task probe black |
| 636 | `349ea2636f33` | 36/500 = 7.2% | others/1101 task066; single-task probe black |

These are network-specific exact-SHA matches.  They do not create a permanent
task066 exclusion.  The cost583 SHA above is different and remains eligible for
an LB probe.

## Artifacts and safety

- `inventory.json`: full SHA-deduplicated discovery inventory
- `result.json`: isolated actual-audit evidence and final decisions
- `probe_manifest.json`: the single remaining probe-only payload
- `winner_manifest.json`: empty fixed-winner manifest
- `evidence/task066_fresh_2seed_500.json`: two-seed dual-ORT fresh results
- `evidence/authority_binding.json`: exact champion binding
- `evidence/workers/`: per-model isolated audit outputs

No ZIP was created or merged.  Root protected files, `others/71403`, and
`submission.zip` were not modified.
