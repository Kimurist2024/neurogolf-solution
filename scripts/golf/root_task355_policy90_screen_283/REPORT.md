# task355 cost249 normal-POLICY90 audit

Decision: **PASS_POLICY90_PRIMARY_AND_INDEPENDENT_REVIEW**.

The immutable authority is `submission_base_8009.46.zip` (SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`).
Its task355 member has SHA-256 `f87a253a7925f7bd848135471b29dea8c0b5c3c2335162f2d7f38ea566bc8a6c`
and cost 250.  The selected candidate is
`task355_r04_static249.onnx`, SHA-256
`7ca617858a19310a433010e6e50da46b4d562d76f3d0688665c8387bdf6f24d8`,
with memory227 + params22 = cost249.  The projected gain is
`ln(250/249) = 0.004008021397538641`.

The graph delta removes the one-byte `WA = WT + WB` tensor and changes only
the `tieval` multiplication from `cnt4 * WA` to `cnt4 * h_num`.  This is not
an exact task-rule proof; it is admitted under the user's normal POLICY90
threshold.

Primary audit:

- known: 264/267 = 98.8764%;
- fresh seeds `283555001` and `283655001`: 9872/10000 and 9852/10000;
- disabled/default ORT with threads1/4: identical signs;
- runtime errors, nonfinite outputs, output-shape mismatches, forbidden small
  positives, Conv-bias UB, lookup/fixture table, giant graph and shape cloak:
  all zero;
- minimum positive output: 1.0.

Independent audit used disjoint seeds `284355001` and `284455001` and obtained
9871/10000 and 9860/10000 in every one of the four ORT configurations.  Across
81,068 executions it observed zero runtime, nonfinite, shape, margin, sign or
raw configuration mismatches.  Every declared intermediate also matched its
runtime shape.

task355 appears in the public overfit-risk discussion but is not in the
project's private-zero catalog.  Therefore this result is explicitly normal
POLICY90, not complete official correctness.  The primary evidence is
`evidence.json`; the independent evidence is under
`scripts/golf/agent_review_task355_policy90_284/`.
