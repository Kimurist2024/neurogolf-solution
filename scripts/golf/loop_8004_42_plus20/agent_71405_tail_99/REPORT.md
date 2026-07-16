# 71405 tail priority audit (lane 99)

## Outcome

The 12 requested `others/71405` candidates were audited against the assigned
authority `submission.zip` SHA-256
`9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`.
All 12 are competition-scorer strict reductions, but there are **zero fixed
winners** because none has exact-SHA LB-white proof.

Three candidates remain isolated LB probes:

| priority | task / candidate | cost | projected gain | runtime shape | fresh | classification |
|---:|---|---:|---:|---|---|---|
| 1 | 310 `task310_improved(1)` | 566 -> 501 | +0.121988 | truthful | 4999/5000 worst mode across two 5000-case seeds | `LB_PROBE_REQUIRED_HIGH_RISK` |
| 2 | 396 `task396_cost_reduced` | 1019 -> 960 | +0.059644 | truthful | 482/500, 489/500 per seed; 971/1000 aggregate | `LB_PROBE_REQUIRED_KNOWN_RISK_LOW` |
| 3 | 396 `task396_improved_v2` | 1019 -> 974 | +0.045166 | truthful | 482/500, 489/500 per seed; 971/1000 aggregate | `LB_PROBE_REQUIRED_KNOWN_RISK_LOW` |

The two task396 models are mutually exclusive.  Selecting task310 plus the
better task396 probe would have a local projected gain of `+0.1816317259`, but
this is **not** an LB estimate and no probe ZIP was created.

## Probe risk and exact history

- task310 SHA
  `4eed21efedf2b44e11d2bb748d383275d193144c3c0f8f9f55265c8639e6fdec`
  has no exact LB record.  Known x4 and direct runtime shapes are clean, but the
  graph uses six `TfIdfVectorizer` nodes and a 31-input Einsum.  Expanded fresh
  found one failure at case 4037 of seed 202607149901 (132 differing cells), so
  fixed adoption is unsafe.  The related cost-566 authority is also locally
  imperfect in the earlier C9 audit.
- task396 SHA
  `75eb6d0696e220bce204281f246a96aabc8e0db976f777555c268c0ae08e67ef`
  and SHA
  `1e6ed65101d408aca7050830716463358d55b72d21b72a30449ae93fa7d6429e`
  have no exact LB records.  Both reproduce the same fresh failures.  Related
  task history is explicitly version-dependent: cost 1026 was LB-white and a
  cheaper cost-982 version was LB-black; task396 is catalogued as a repeated
  private-zero task.  These are low-priority, high-risk probes only.

The exact SHA searches found only the new `root_71405_96` inventory records,
not a prior LB result for any of these 12 exact payloads.

## Rejections

| task / candidate | cost | known x4 | fresh worst | terminal reason |
|---|---:|---|---|---|
| 354 `task354_improved` | 537 -> 536 | pass | 500/500 | 7 runtime/declaration shape mismatches |
| 361 `task361_cost844` | 858 -> 844 | fail | not run | default ORT rejects malformed CenterCropPad shape; task has black-version history |
| 363 `task363_improved` | 513 -> 512 | pass | 495/500 | 7 shape mismatches plus fresh failures |
| 365 `task365_cost1355` | 1369 -> 1355 | pass | 500/500 | 12 shape mismatches; another task365 SHA is directly LB-black |
| 370 `task370_improved(1)` | 954 -> 944 | fail | not run | default ORT rejects inconsistent Concat shape |
| 378 `task378_improved` | 525 -> 522 | pass | 490/500 | direct trace hits ORT buffer shape-reuse error; fresh failures |
| 268 `task268_improved_cost420` | 422 -> 420 | fail | not run | default ORT failure, 1,476,925-byte oversize lookup graph |
| 270 `task270_improved` | 594 -> 587 | pass | 500/500 | 4 shape mismatches and giant Einsum |
| 284 `task284_improved` | 518 -> 517 | pass | 500/500 | 11 shape mismatches |

Fresh was deliberately not run for tasks 361, 370, and 268 because their
default-ORT known gate failed.  Passing disable-all alone cannot rescue a
candidate that the default competition runtime cannot construct.

All candidates pass ONNX full checker, strict shape inference, standard-domain,
zero-function, finite-initializer, and Conv-bias-UB checks except for the
candidate-specific structural failures recorded above.  Generic strict shape
inference does not override the direct runtime-shape witnesses.

## Tail/order and workspace safety

Tasks 354, 363, 370, and 378 were evaluated only as isolated replacements.
No ZIP was built, no tail member was reordered, and this lane did not modify
root files or `others/`.

During finalization another shared lane changed the live root `submission.zip`
from the assigned SHA `9085e2...` to SHA
`50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6`.
All evidence here was already captured against the assigned `9085e2...`
authority and was not silently rebased.  The drift is recorded in
`result.json` and `audit/deep_audit.json`.

## Evidence

- `audit/deep_audit.json`: combined competition profiles, known x4, static,
  runtime-shape, fresh, exact-history, and decisions
- `audit/rows/*.json`: subprocess-isolated per-candidate raw evidence
- `probe_manifest.json`: three non-fixed probe candidates
- `winner_manifest.json`: empty fixed-winner list
- `result.json`: lane summary and authority-drift record
- `audit_tail.py`, `finalize.py`: reproducible audit and assembly scripts
