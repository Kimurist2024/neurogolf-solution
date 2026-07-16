# task192 duplicate-constant regolf 170

## Result

Accepted SOUND candidate: `task192_dedupe_values.onnx`, SHA-256
`51a7d65491f300b91243ef8b522c00d2b699eb07615a5d64c5694de33e4d0490`.

- Immutable 8009.46 authority: cost **1609**.
- Previous staged exact candidate: cost **1197**.
- New staged exact candidate: cost **1195** (memory248, parameters947).
- Authority-relative projected gain: **+0.2974666826267721**.

The previous graph stored two byte-identical float32 tensors,
`hist_select=[0,1]` and `onehot_values=[0,1]`. The OneHot node now consumes
`hist_select` and the duplicate initializer is removed. This is an all-input
identity: every consumer receives the same dtype, shape, and values, while the
graph computation and evaluation order are otherwise unchanged.

Full checker, strict data propagation, static/truthful shapes, standard-domain,
Conv-UB0, finite-initializer, and no-lookup gates pass. Known265/265 passes in
all four ORT/thread configurations. Two fresh seeds pass5000/5000 each with
zero runtime/nonfinite failures and the complete163-case sign proof remains
valid. Evidence: `dedupe_build.json`, `audit/task192_exact_poly.json`.

The exploratory ReduceSum+Slice histogram rewrite cost1245 and is rejected.
The accepted duplicate-constant SHA replaces only `others/71407/task192.onnx`;
root submission and score ledgers remain unchanged.
