# task251 runtime-safe repair — rejected

## Outcome

There is no runtime-safe strict-lower candidate. Nothing should be staged or
promoted; verified gain is **0.0**.

The archive r03 model's apparent cost 582 depends on 58 false declared shapes.
Repairing all malformed `CenterCropPad` shape inputs and declaring the actual
runtime shapes produces a clean model, but its truthful cost is **295,949**
(memory 295,713 + params 236), versus authority cost 755. The repaired model's
score delta versus authority would be -5.971225, so it is rejected immediately.

| model | checker + strict data-prop | default ORT | shape mismatches | cost | decision |
|---|---:|---:|---:|---:|---|
| r03 archive | pass only because shape values stay opaque | session fails | 58 | 582 apparent | reject |
| r03 arity-only diagnostic | fail | train[0] passes | retained cloak | unscorable | reject |
| r03 truthful repair | pass | train[0] passes | 0 | 295,949 | reject |

The truthful rejected artifact is
`task251_r03_arity_repaired_truthful.onnx`, SHA-256
`fad302b70f5e9b9243ee922dfbbbd5e86ed8cae39e8c854f065d15f8576a7c50`.
It is evidence, not a candidate for submission.

## Exact failure and repair

`CenterCropPad` requires one target size per axis. r03 feeds a one-element
shape tensor to 52 multi-axis calls:

- `[1,2,3]`: 5 calls; scalar 30 once, 29 twice, 28 twice.
- `[2,3]`: 47 calls; scalar 30 sixteen times, 29 sixteen times, 28 fourteen
  times, and 12 once.

The runtime with optimizations disabled broadcasts those scalars. The exact
legal repair is therefore `[n,n,n]` for the three-axis calls and `[n,n]` for
the two-axis calls. This makes default ORT executable. Keeping the old
`[1,1,1,1]` value_info then fails full checker/strict inference, correctly
exposing the cloak. Replacing all 64 intermediate declarations with traced
runtime shapes yields:

- full checker and strict shape inference with data propagation: pass;
- static positive shapes, standard domains, finite initializers, banned-op /
  nested-graph / function / sparse / external-initializer checks: pass;
- direct declared/runtime shape audit: 0 mismatches;
- Conv/QLinearConv bias UB check: 0 findings;
- ORT disabled and default train[0] probe: exact gold, errors 0.

The first intermediate alone is `ch30a: float32[1,30,30,30]` = 108,000
bytes. Thus no metadata repair of this family can fit below cost 755. r04 has
the same legal `ch30a` hard floor; its 52 malformed calls are all spatial
`axes=[2,3]` (30:17, 29:18, 28:16, 12:1), so emitting another above-ceiling
repair cannot change the decision.

## Validation scope and safety

Complete known-corpus and two-seed/four-configuration fresh validation are
admission gates for a strict-lower candidate. They were not run after the
truthful cost gate failed by 295,194. There is no new error task and no
private-zero pass-through claim: no model is promoted.

The authority remains cost 755, SHA-256
`57f557717f6c9b582b0051519e722721bc5c904fd310252bcd4030f2df8d5c63`.
No root submission, score file, staging area, or `others/71407` artifact was
written. `try_candidate.py` and Kimi were not invoked.

Machine-readable evidence is in `result.json`; the deterministic repair and
shape-tracing procedure is in `repair_center_crop_pad.py`.
