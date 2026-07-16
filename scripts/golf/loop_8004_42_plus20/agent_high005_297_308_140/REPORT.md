# task005 / task297 / task308 exact-regolf lane (8009.46 authority)

## Outcome

No candidate survived every fail-closed gate. Projected accepted gain is
`0.0`; `winner_manifest.json` is intentionally empty. Nothing from this lane
was merged into a root model, score CSV, or submission ZIP.

| task | immutable cost | best lower probe | decision |
|---:|---:|---:|:---|
| 005 | 2325 | none | no strictly-lower exact/clean candidate |
| 297 | 371 | 370 | rejected: 1/265 known-correct |
| 308 | 433 | not costable | rejected: shape cloak and ORT session failure |

The immutable authority is `submission_base_8009.46.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.

## task005

The authority member is SHA-256
`77eb35fdcf2dbbacaa1c63d2dfef4f3b50ecbfbc8178da3bc2e7883ee8275c57`
and costs 2325. Its baseline passes checker, strict data-propagating shape
inference, both ORT modes on all 266 known cases, and has zero runtime shape
mismatches. It nevertheless contains a Hardmax selector and is known to share
duplicate-guide-color failures on generated data, so an authority-equivalent
micro-regolf would not meet this lane's clean true-rule requirement.

The same member SHA was already exhaustively inventoried in lane B14: 113
deduplicated task005 models / 970 raw sources and all 48 installed optimizer
passes. The lowest non-base static floor was a 2325 tie. This lane's additional
cleanup, initializer-reuse, CSE, attribute, and route-free construction review
found no strictly-lower shape-preserving candidate. Removing the 98-element
route table requires dynamic row/column carriers whose charged intermediates
exceed the saved parameters.

Two controls confirm the current trade-off:

- the known/fresh-sound rebuild costs 2389 (+64) and still uses Hardmax;
- the clean true-rule control costs 2545 (+220).

Neither is a score improvement, so task005 has no winner.

## task297

The authority member costs 371 and is 265/265 known-correct in both ORT modes,
with full checker/strict inference and zero runtime shape mismatches.

The only new strict-lower probe removes `q_scale`, shares `c_scale`, and changes
the two hash coefficients from `[5,9]` to `[34,61]`. It costs 370 and is
structurally clean, but the apparent real-valued rescaling identity is false in
the actual graph: `Mul` operates on uint8, so products such as `34*10` and
`61*10` wrap modulo 256. The reachable code 10 therefore changes quantized
features from `[4,6]` to `[1,1]`. Both ORT modes score it 1/265, errors 0.

The historical 361-cost zero-kernel trim is also inadmissible because it uses
negative Conv padding `[0,0,0,-24]`, contrary to the ONNX Conv schema.
Schema-compliant Slice and Split/re-concat variants cost 484 and 511. No
task297 candidate reached the fresh gate.

## task308

The authority member's nominal cost is 433 only under contradictory declared
shapes. Runtime tracing finds 26 declared/actual mismatches and 63,546 bytes of
actual single-example intermediates versus 376 nominal memory. Default ORT
cannot create a session because TopK requests more elements than its inferred
axis contains. Preserving this shape cloak is outside the allowed optimization
domain.

Bypassing an exact identity copy of a fixed four-element shape vector removes
two nodes locally, but the candidate still fails full/strict inference and both
ORT session modes at TopK, and cannot be costed. A truthful repair necessarily
exposes the large runtime tensors and is not a strict-lower regolf. No task308
candidate reached known or fresh admission.

## Validation policy

Fresh validation was deliberately not run: every candidate failed an earlier
mandatory gate (clean structure/admissible semantics, strict-lower actual cost,
known dual-ORT, or authority equivalence). Running fresh cases cannot repair a
known semantic failure or an unloadable graph. `candidate_audit.json` contains
the mechanical evidence, and `finalize.py` rechecks immutable hashes and the
fail-closed decision.
