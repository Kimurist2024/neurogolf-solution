# Lane C41 — task204 exact-current re-audit

## Verdict

**NO WINNER; projected gain +0.0.**  No model is eligible to replace task204 in
`submission_base_8002.63.zip`.  The archive and every shared score/submission
artifact remain unchanged.

The exact authority is archive SHA-256
`a2da30657f3798e861f369ac896f36722ff658ed3e468c4d55db9a04eefbccfc`.
Its task204 member is zero-based position 370, SHA-256
`7405b5d8560f0bbd4626c5ca6a1c416d21947d3bb9572941b6b37ff10d39ad88`,
and actual official-like cost **2240** (`memory=2084`, `params=156`).

## Why the apparent 2240 floor cannot be shaved safely

The current member is correct on all 268 known examples with
`ORT_DISABLE_ALL`, but it is not a safe candidate template:

- declared output is `[1,10,1,2]`, while runtime output is `[1,10,30,30]`;
- an all-intermediate trace finds **53 declared/actual shape mismatches**;
- default ORT cannot even create a session: the `Concat` shape merge sees
  inferred dimension 1 versus declared dimension 20;
- task204 is explicitly catalogued in `docs/golf/private_zero_tasks.md` as an
  order/cross-task contamination network, and the older audit requires keeping
  its archive position fixed.

Thus further value-info/output shrinking, shape cloaking, or moving the member
would violate this lane's acceptance contract even if it lowered the scorer's
reported memory.

## Historical search

The prior rebuild winner (`6e21b4...`) is **not** the current member.  Its cost
is 2544, already 304 worse than the current 2240; it declares output
`[1,10,1,30]`, has 57 runtime-shape mismatches, and fails default ORT session
creation.  The complete eight-SHA archive inventory measures costs
`2705, 2706, 2779, 2791, 2860, 2544, 2980, 3502`.  Every one is more expensive
than the exact current member, and every one declares a non-truthful output
shape.  No archived/private-zero candidate is reusable.

The earlier isolated pollution audit reached the same operational conclusion:
safe task204 variants were more expensive, while `CastLike`/`CenterCropPad`
allocator-risk variants had to remain quarantined and position-stable.

## New exact-current experiment

`build_truthful_candidate.py` removes the full-grid integer cast and rebuilds
all value-info/output metadata with observed truthful shapes.  The resulting
`task204_truthful_floatpack.onnx` passes full checking and strict/data-propagating
shape inference with **zero** runtime-shape mismatches.  It is rejected before
fresh testing because it costs **6200** and is wrong on all 268 known examples
in both ORT modes.  It demonstrates why correcting the incumbent's hidden
large intermediates does not produce a cheaper candidate.

Fresh/dual-5000 was intentionally not spent: no candidate survived the cheaper,
known-complete, two-ORT, and truthful-runtime-shape gates.  Detailed hashes and
measurements are in `audit_summary.json`.
