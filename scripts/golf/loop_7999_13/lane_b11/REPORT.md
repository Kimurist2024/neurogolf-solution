# Lane B11 — exact-base 7999.13 audit

## Result

No candidate was adopted. Aggregate cost delta is `0`; aggregate leaderboard-score delta is `+0.000000`.

The exact source was `submission_base_7999.13.zip`, SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
No root ZIP, CSV, score ledger, or optimized artifact was changed.

## Scope and method

Tasks `264, 281, 300, 358, 376, 387, 392` were extracted directly from the exact ZIP and rescored. Their generator rules were classified before optimization. Every locally associated ONNX was then SHA-256 deduplicated and screened: 300 unique models total. Static parameter plus declared-memory lower bounds removed models that could not beat the exact member; every remaining distinct alternative received the applicable checker, structural, and dual-ORT early gates.

Adoption required all known examples, independent seed-5000 fresh verification with 5000 cases in each of `ORT_DISABLE_ALL` and default ORT, zero errors, full checker, strict inference, default domains, truthful runtime shapes, and no giant Einsum, lookup representation, metadata-only saving, or undefined behavior. No alternative survived the earlier gates, so none was eligible for the expensive fresh-5000 adoption run.

## Per-task disposition

| task | exact cost | local unique | lower-bound alternatives | result |
|---:|---:|---:|---:|---|
| 264 | 362 | 56 | 1 | Reject. Exact member has 44 runtime shape mismatches and fails default ORT. The cost-358 history candidate fails its first known case in both ORT modes. |
| 281 | 161 | 79 | 0 | No candidate. Exact member has two runtime shape mismatches and a 38-input giant `Einsum`. |
| 300 | 182 | 28 | 0 | No alternative. The sole row below the declared lower-bound threshold is the exact member itself; it has seven runtime shape mismatches and 20/25/26-input `Einsum` nodes. |
| 358 | 161 | 41 | 0 | No candidate. Exact member is runtime-shape truthful but uses a 44-input giant `Einsum`. |
| 376 | 158 | 14 | 1 | No adoption. Exact member is clean. The nominal cost-65 uint8-index variant fails full checker because `Gather` requires int32/int64 indices. The valid architecture already pays the 120-byte int32 `[30]` row-index floor, two 4-byte scalars, and 30 selector parameters: 158 total. |
| 387 | 337 | 57 | 0 | No candidate. Exact member has 16 runtime shape mismatches; no local model has a declared lower bound below it. |
| 392 | 345 | 25 | 5 | Reject. All five distinct lower-bound variants use prohibited `TfIdfVectorizer` lookup encoding and fail the first known case in both ORT modes. |

## Evidence

- `baseline_inventory.json`: exact extraction hashes and official-like costs.
- `exact_graph_audit.json`: node/initializer/consumer inventory and duplicate checks.
- `baseline_shape_safety.json`: checker, strict inference, actual-vs-declared runtime shapes, and one-case dual ORT.
- `history_scan.json`: all-local SHA-deduplicated lower-bound screen.
- `candidate_rejections.json`: reproducible checker and dual-ORT rejection evidence.
- `manifest.json`: machine-readable final decision and zero-adoption manifest.

The only clean exact member in this lane is task376, and its known local cheaper attempt is invalid at the ONNX type gate. Reopening this lane would require a genuinely different direct-output construction for task376 or new spec-derived architectures for the other tasks; local pruning and harvesting are exhausted under the requested safety policy.
