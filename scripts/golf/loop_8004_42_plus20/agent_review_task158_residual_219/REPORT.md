# task158 residual candidate — independent review 219

## Decision: PASS

The candidate is a SOUND, strictly cheaper, all-input-equivalent rewrite of the
staged task158 authority over the complete reachable anchor-score support.

- parent: `others/71407/task158.onnx`
- parent SHA-256: `127984c6807d84559bbf74fd58e3b09a66459d142cef65a8635647e64f5e59fd`
- candidate: `scripts/golf/loop_8004_42_plus20/agent_task158_residual_215/candidates/task158_exact_anchor_role_bits.onnx`
- candidate SHA-256: `e7101699bfc022fa794e15d7f374a8febe3e2680b8388c67b9a81cdc9962ced0`
- official profile: **7525 -> 7498** (`memory 6662 -> 6638`, `params 863 -> 860`)
- cost reduction: **27**
- projected score gain: **+0.003594492321219123**

No root submission, score ledger, or file under `others/71407` was changed.

## Static and official-profile gates

Both parent and candidate independently pass:

- ONNX full checker;
- strict shape inference, with and without data propagation;
- one canonical static input and output, all inferred dimensions positive and static;
- standard ONNX domains only;
- no banned/Sequence ops, functions, nested graphs, sparse initializers, or external data;
- finite initializers;
- Conv/ConvTranspose bias-length UB findings: **0**;
- `Hardmax` / `TfIdfVectorizer`: **0**;
- runtime declared/actual intermediate-shape mismatches: **0**;
- runtime intermediate nonfinite values: **0**.

Runtime tracing covered 169 parent graph outputs and 167 candidate graph outputs.
The competition scorer independently reproduced cost 7525 and 7498 and marked
both complete known corpora correct.

The graph-delta whitelist also passes. No unrelated computation changed:

- removed outputs: `more_role_low`, `more_role_mid`, `more_role_u8`, `role_threshold`;
- added outputs: `anchor_score_u8`, `anchor_low_bits`;
- changed outputs only: `phase_ge_0`, `low_mask`, `anchor_high`;
- removed parameters: `phase_cut_0`, `more_role_1`, `more_role_2`, `more_role_3`;
- added parameter: scalar `anchor_low_bit_mask = 10` (`0b1010`);
- every common initializer is bitwise unchanged.

The candidate does not add a learned or example-indexed lookup. Its only new
parameter is the one-byte role bitmask justified by the complete support proof.

## Independent complete-support proof

The generator has only these local anchor degrees of freedom:

- magnification 1, 2, or 3;
- the two opposite-corner diagonals induced by horizontal/vertical flips;
- either geometric endpoint may carry the numerically lower endpoint colour;
- row and column translation modulo Conv stride 2.

These give **48** configurations. The review enumerates them on an infinite
stride-2 lattice, so finite grid boundaries merely remove windows and cannot
introduce a score. `endpoint_code` is zero on fill and background channels.
The generator's `spacing=2` rule separates distinct boxes by at least three
coordinates on one axis, while a 3-cell Conv window spans only two, so one
window cannot mix endpoints from different boxes.

The independently recovered positive support is exactly:

`{2, 4, 8, 10, 16, 20, 24, 26, 48, 52, 72, 106, 144, 212}`.

At most 9 sampled windows touch one magnified endpoint block. With at most four
objects and two endpoints per object, at most 72 of the 169 Conv windows are
affected. Therefore at least **97 windows are exact zero**, so TopK(8) can never
admit a negative score. Its complete support is the set above plus zero.

For every supported value:

- the parent's low-role predicate equals `(uint8(score) & 0b1010) > 0`;
- the parent's high-role predicate equals `valid XOR low_role`;
- the parent's `score >= 6` phase predicate equals `uint8(score) > 4`.

Thus the role and phase rewrites are exact on every generator-reachable input,
not merely on sampled cases.

## Independent runtime matrix

Known corpus: all **266** cases in each of four configurations:

- ORT_DISABLE_ALL, threads 1;
- ORT_DISABLE_ALL, threads 4;
- default optimizations, threads 1;
- default optimizations, threads 4.

Fresh corpus used new seeds `15821901` and `15821902`, 1,500 cases per seed,
again in all four configurations. Totals:

- known raw comparisons: **1,064**;
- fresh raw comparisons: **12,000**;
- candidate/parent raw bitwise mismatches: **0**;
- parent truth mismatches: **0**;
- candidate truth mismatches: **0**;
- runtime errors: **0**;
- nonfinite values: **0**;
- candidate cells in the unsafe open interval `(0, 0.25)`: **0**.

## Evidence

- `static_math_audit.py`
- `runtime_audit.py`
- `evidence/static_math_profile.json`
- `evidence/support_proof.json`
- `evidence/runtime_four_config.json`

Root guards observed after review:

- `submission.zip`: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `all_scores.csv`: `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`

