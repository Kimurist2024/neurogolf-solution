# Expansion lane 94 — incremental 20-task audit

## Outcome

- Immutable authority: `submission.zip`, SHA-256 `9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118` (LB 8006.61).
- Targets: 102, 025, 250, 062, 324, 308, 008, 275, 338, 333, 268, 184, 377, 109, 160, 099, 279, 345, 170, 245.
- Fixed winners: **0**. LB probes: **0**. Protected root files and `others/` were not modified.
- The predecessor scans already covered **930** task×SHA candidates. The current repository has **936**, with only **7** genuinely new SHAs.

## Incremental inventory

The scan observed 12,362 matching loose ONNX files and 23,843 target members from 1,269 ZIPs. New SHAs were one each for tasks 025/062/170/245/308/338/377. Task333 had no delta because its dedicated 41-SHA audit was used as a predecessor inventory.

One expected inventory diagnostic was recorded: `submission_base_8006.61.zip` is the known 68-byte drifted JSON, not a ZIP, and was never used as authority.

## Seven new candidates

| task | authority | candidate | nominal gain | SHA | decision |
|---:|---:|---:|---:|:---|:---|
| 025 | 474 | 474 | 0.000000 | `2f5cf0ae7ba8` | REJECT_NOT_STRICTLY_LOWER |
| 062 | 465 | 463 | 0.004310 | `6767dbf75899` | HARD_REJECT_SHAPE_CLOAK |
| 170 | 387 | 384 | 0.007782 | `e5fea4c41d22` | HARD_REJECT_STRUCTURE_OR_UB |
| 245 | 387 | 385 | 0.005181 | `228b6ad9f245` | HARD_REJECT_STRUCTURE_OR_UB |
| 308 | 434 | 433 | 0.002307 | `fc845e9edee0` | HARD_REJECT_RUNTIME_CONFIG |
| 338 | 426 | 406 | 0.048086 | `09e8436ab305` | HARD_REJECT_RUNTIME_CONFIG |
| 377 | 409 | — | 0.000000 | `a0e952e93ee0` | HARD_REJECT_OFFICIAL_RUNTIME |

Competition `score_and_verify` was repeated three times with independent labels/tempdirs for tasks 062/170/245/308/338; authority and candidate profiles were identical on every run. This corrects task170's old shape-dependent 357 diagnostic: the competition authority cost is 387 (matching `all_scores.csv`) and the candidate is 384, but the candidate still fails strict data-propagating shape inference.

## Why nothing is probeable

- task062 is known-perfect 267/267 in all four ORT/thread configurations, but three intermediates declare `[1,1,1,1]` while runtime is `[1,10,30,30]`; the 2-point shave is a shape cloak.
- task170 and task245 fail strict data-propagating ONNX shape inference.
- task308 is known-perfect only with optimizations disabled; default ORT cannot load its TopK graph.
- task338 gives the largest nominal reduction, 426→406, and is known-perfect with optimizations disabled. Default ORT rejects its output Concat shape, and the graph carries pervasive `[1,1,1,1]` declarations for runtime-spatial tensors. It is not a truthful/runtime-safe probe.
- task377 uses an unsupported TopK(uint8) path and cannot be official-profiled. task025 is equal-cost.

Fresh 2×500 was intentionally not run: no candidate cleared schema/UB, known×4, and truthful-shape gates. Fresh is a prioritizer, not permission to bypass those hard gates. Exact-SHA search found no LB record for any of the seven new SHAs.

## task297 legal-repair pivot

The known/fresh-perfect cost361 candidate relies on `Conv pads=[0,0,0,-24]`. Existing legal Slice and Split rewrites cost484 and511. A legal stride-1 Conv producing width6 needs a width25 kernel (+240 parameters), while higher stride samples spaced rather than contiguous columns. No schema-compliant repair at cost≤370 was found; the authority cost371 remains.

## Artifacts

- `inventory_delta.json`: full incremental SHA/source inventory.
- `audit/incremental_screen.json`: official/known×4/shape/UB decisions.
- `audit/official_reprofile_3x.json`: independent competition-profile repeats.
- `audit/exact_sha_lb_history.json`: exact-SHA history result.
- `audit/task297_legal_repair_analysis.json`: legal rewrite bounds.
- `result.json`, `probe_manifest.json`, `winner_manifest.json`: final disposition.
