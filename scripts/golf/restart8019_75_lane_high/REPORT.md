# 8019.75 high-cost/new-drop lane

- Immutable authority: `submission_base_8019.75.zip`
- SHA256: `e69058edd21e27ab7d32670d714ec5cea6d35632a9d9a620364731297717edb3`
- Scope: all 123 authority tasks whose ledger cost is at least 300, excluding
  task275 because it was already staged by another lane.
- Sources: every loose ONNX and ZIP member in `others/71604` and
  `others/71605`, compared against reprofiled authority cost.
- Admission gate: local gold exact + official gold exact, static/checker safe,
  stable positive margin, fresh 2,000 x 2 at 100%, and zero runtime errors,
  nonfinite values, shape mismatches, or small-positive values.

## Admitted candidates

| task | authority -> candidate | projected gain | known | fresh | raw identity | classification |
|---:|---:|---:|---:|---:|---|---|
| 132 | 308 -> 292 | +0.053346 | exact | 2000/2000 x2 | no | strict-policy probe only |
| 168 | 398 -> 384 | +0.035809 | exact | 2000/2000 x2 | yes | constant initializer reconstruction |
| 226 | 369 -> 368 | +0.002714 | exact | 2000/2000 x2 | yes | redundant `Where` removal |
| 345 | 389 -> 369 | +0.052783 | exact | 2000/2000 x2 | yes | constant initializer reconstruction |

All four also passed official gold exact and the final strict gate.  The raw
identity column compares the unthresholded output against the 8019.75 authority
on every known train/test/arc-gen input.  task168 and task345 replace literal
constant tensors with input-value-independent `PRelu` + `CastLike`
reconstruction; the downstream graph is unchanged.

## Ready-to-submit evidence bundles

- Safer exact-three bundle: `submission_EXACT3_tasks168_226_345.zip`
  - SHA256 `42bf550374bf5dd57ec99979e348a1ad4399493040240fbd8f1d7d001339bf5d`
  - only tasks 168, 226, and 345 differ
  - projected +0.091306, or 8019.841306 from 8019.75
- Strict-four bundle: `submission_STRICT4_tasks132_168_226_345.zip`
  - SHA256 `0f45d2a08f1672c2966efbf833ef904e0cdf4324114e7ce980aa073a77c06bca`
  - projected +0.144652, or 8019.894652 from 8019.75
  - task132 is not raw-bit-identical, so retain probe-level caution
- Individual probes exist for tasks 132, 168, 226, and 345.  Exact paths and
  hashes are in `exact_probe_manifest.json`.

## Rejected or quarantined screen survivors

- task198@595: rejected; both fresh streams were 1999/2000.
- task310@332: rejected; both fresh streams were 1999/2000.
- task157@826: not admitted; task157 has a confirmed private-grader crash
  history that local/fresh testing cannot reproduce.
- task333@409: not admitted; full fresh evaluation is exceptionally slow and
  task333 has prior unresolved private-zero ambiguity with task198.  It is not
  appropriate for the deadline bundle.

The fast census found no other known-exact, strictly cheaper candidates among
the 123 tasks.  Additional focused checks of the highest projected-gain loose
files (13, 99, 165, 182, 238, 239, 270, 297, 308, 354, 367, 368, 370, 387,
398) all failed before admission, usually because the 8019.75 authority already
contained an equal/better model or because official/strict correctness failed.

## Root protection

No root authority was updated by this lane. `submission.zip`,
`all_scores.csv`, and `best_score.json` had identical SHA256 values before and
after every bundle build.  See `exact_probe_manifest.json` for the recorded
guards.
