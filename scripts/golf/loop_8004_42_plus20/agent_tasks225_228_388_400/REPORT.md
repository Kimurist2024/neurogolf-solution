# Tasks 225 / 228 / 388 / 400 — latest-authority SOUND exact re-golf

## Result

`winner: null` — no strict-lower candidate survives the official, runtime-shape-aware cost gate.

The immutable authority is `submission_base_8009.46.zip`.  Its task payload SHA-256 and independently reproduced official profiles are:

| task | SHA-256 | memory | params | cost |
|---:|---|---:|---:|---:|
| 225 | `c55b5673a1e36b07114e82a629d23b01cefbd7b56289ad314b272d7180ef8a4a` | 233 | 100 | 333 |
| 228 | `13946c8d5a52886212f495b13fa6c128a091e77b84e10e27b442ed87f9694a45` | 241 | 50 | 291 |
| 388 | `f27fa5f4f0bcade23d02fed2a74e3c2b826b11140bd03d29f47e0c59c382a8e1` | 283 | 22 | 305 |
| 400 | `89b419dbad732d3235ac1ab7d078ef22eef3209eb8b5f30e21d3a502ccd03389` | 123 | 41 | 164 |

No root submission, ledger, or `others/` artifact was changed.

## Generator rules audited

- task225: fixed 6x6 grid.  A 2x2 four-color seed is retained; each corresponding color is stamped as a clipped 2x2 block at the four diagonal offsets `(±2,±2)`.  The authority encodes this finite geometry, not an example table.
- task228: a 4..6 by 4..6 colored rectangular border contains four colored inner-corner pixels.  Those pixels are erased and moved to the four diagonally outside corners with 180-degree/opposite-corner correspondence.
- task388: duplicate the source tile 2x2; every source column containing a colored pixel becomes cyan (8) throughout, while the original colored pixels retain their source color.
- task400: recover the generator's 5x5 cutout from its blue (1) markers inside the 24x24 eightfold-symmetric pattern.  The compact Sakana expression matches all 266 stored cases but has a rare duplicate-row ambiguity on 4/2000 fresh generator cases; the generator itself is therefore the authority.  The ONNX payload matches the generator output on all 2000 fresh cases.

## Correctness and safety gates

For every task:

- full ONNX checker and strict shape inference with `data_prop=True`: pass;
- banned ops, nested graphs, functions, nonstandard domains, sparse initializers: zero;
- `Hardmax`, `TfIdfVectorizer`, giant initializer, malformed Conv/ConvTranspose bias: zero;
- stored train+test+arc-gen under `ORT_DISABLE_ALL`, thread 1: task225 `265/265`; other tasks `266/266`;
- fresh generator: two independent seeds x 1000 cases x four configurations (`DISABLE_ALL/default` x threads `1/4`), every task `8000/8000` decoded correct;
- raw outputs are bit-identical to the `DISABLE_ALL/thread1` reference in all `8000/8000` fresh configuration runs per task;
- runtime errors and non-finite outputs: zero.

Detailed results are in [`evidence/authority_audit.json`](evidence/authority_audit.json).

## Runtime-shape gate

The audit exposed every node output on a fresh input and compared the declared/inferred shape with the actual ORT shape:

- task225: 25/25 observed outputs truthful.
- task228: the existing authority has 8 data-dependent shape mismatches around its 16-row ScatterND index construction.
- task388: the existing authority has 14 mismatches, including the 1x10x30x30 carrier hidden behind 1x1x1x1 type templates.
- task400: the existing authority has 4 mismatches in its GroupNormalization / log-reduction carrier.

This explains why zero-input/static profiling is unsafe here.  All promotion decisions used the full known corpus and official profiler, which takes the maximum runtime shape.  No new shape mismatch was admitted.

## Search and rejected candidates

- 96 fixed-point optimizer profiles across 24 exact pass sets: `strict_lower=0`.
- task388 `CastLike(q,bool_ref) -> Greater(q,8)` is exact on complete generator support (`q ∈ {0,248}`), passes checker and known gold, and removes one parameter.  Official cost becomes **9303**, because `Greater` exposes the real 1x10x30x30 shape; reject.
- task388 `CastLike(ids_hid,i8_ref) -> Cast(to=int8)` is exact and known-correct but official cost becomes **1599**; reject.
- moving the uint8 reduction before the bool conversion cannot remove the full boolean background mask: the same mask is required as the second `Where` condition.  The attempted graph is invalid; reject.
- task225's 5x5 routing initializer cannot be reduced to 4x4 with a single nearest-neighbor Resize: its required separable sampling partition is `[0,0,1,2,3,3]`, which no in-bounds 4-to-6 ONNX nearest mapping produces.  Generating sparse routing values with nodes costs more intermediate memory than the saved parameter count.
- task228's 30-entry coordinate carrier and 16-entry ScatterND updates are already initializer-free at runtime; constructing them in-graph costs at least their full output tensor memory, exceeding any parameter saving.
- task400's remaining tensors are actively consumed and have no exact aliases.  Replacing its type/shape carriers similarly exposes large runtime shapes.

Rejected-candidate profiles are in [`evidence/manual_attempts.json`](evidence/manual_attempts.json), and the mechanical scan is in [`evidence/optimizer_scan.json`](evidence/optimizer_scan.json).

## Decision

Keep all four latest-authority payloads unchanged.  The safe gain from this lane is `+0.000000`; all apparent static reductions either fail semantics/schema or lose under the truthful official runtime profile.
