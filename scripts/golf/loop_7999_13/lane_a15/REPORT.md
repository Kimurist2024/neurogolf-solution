# A15 strict cost audit

## Outcome

- Exact source: `submission_base_7999.13.zip`
- Source SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Tasks: 170, 234, 239, 333, 338, 374, 377
- Retained candidates: 18
- Exact-byte-distinct loose historical models: 247
- Safe winners: 0
- Verified gain: `+0.000000`

No root ZIP, CSV, score pointer, or ledger was modified by this lane.

## Retained result

| Task | Exact cost | Result |
|---:|---:|---|
| 170 | 387 | No retained lead; all loose-history alternatives are non-cheaper or structurally rejected. |
| 234 | 368 | Known-correct lead actually costs 441. |
| 239 | 384 | Cost 374/379 leads fail known data; the other lead is unscorable. |
| 333 | 449 | Both leads inherit 40-47 input giant Einsums. |
| 338 | 426 | Cost 424 fails known/runtime; known-correct lead costs 18423; the remaining lead is giant. |
| 374 | 481 | Known-correct lead costs 876. |
| 377 | 409 | Known-correct leads cost 431-481; cost 408/409 leads fail known data. |

No candidate is both strictly cheaper, complete-known correct, and structurally
eligible. Fresh 5000/5000 was therefore not run.

## Full loose history

| Task | Unique models | Result |
|---:|---:|---|
| 170 | 37 | 13 sound screens not cheaper; 23 structural rejects. |
| 234 | 42 | All 41 alternatives screen at or above cost 368. |
| 239 | 23 | Two cheaper full profiles fail known; one candidate is unscorable. |
| 333 | 25 | Nine sound screens not cheaper; 15 structural rejects. |
| 338 | 36 | 26 sound screens not cheaper; four structural and five runtime rejects. |
| 374 | 26 | All 25 alternatives screen at or above cost 481. |
| 377 | 58 | Two cheaper full profiles fail known; four are unscorable; 51 are not cheaper. |

The archive inventory additionally covers 1,195 ZIPs, 224,111 ZIP members,
and 118,938 loose observations globally. Exact hashes, paths, structure
findings, screen costs, and full profiles are in `loose_history_scan.json`.

## Current-model and prior exact analysis

All exact members pass ONNX full checking, use standard domains, have no Conv
bias finding, and contain no unused or identical same-shape initializer pair.

- task170 was independently fresh-validated 5000/5000 in the earlier C5 sound
  audit. Its fixed-shape scaffolding cannot be replaced by a literal without
  exposing hidden tensors; strict shape inference with data propagation also
  records existing Reshape declarations of length one where length three is
  inferred. No new candidate improves on cost 387.
- task234's reversed coefficient rows cannot be shared without a charged axis
  reversal; the retained candidate costs 441.
- task239's inactive-feature and sentinel removals are exactly the cost-374 and
  cost-379 known failures already identified by B5.
- task333's apparent prefix sharing requires Slice outputs whose cost lower
  bound is 659, above the exact 449. Retained models additionally violate the
  giant-Einsum gate. Given its private-zero/branch ambiguity history, no
  known-only candidate was allowed to advance.
- task338's Boolean fusion triggers the incumbent shape-reuse mismatch, while
  removing its CastLike anchor exposes a cost-18423 tensor. Neither is safe.
- task374's CastLike-to-Cast rewrite is semantically correct but exposes a full
  10x10 tensor and raises cost to 876.
- task377's exact CSE variants expose larger real profiler shapes and cost at
  least 431. Empty-TopK shortcuts are schema-invalid.

## Admission disposition

Fresh validation cannot rescue a known-wrong, costlier, unscorable, giant-
Einsum, metadata-dependent, or UB candidate. Since the prerequisite set is
empty, `winners` remains empty in `final_manifest.json`.
