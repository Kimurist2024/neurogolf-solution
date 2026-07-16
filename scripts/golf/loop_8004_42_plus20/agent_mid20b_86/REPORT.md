# Expanded mid-cost 20-task audit — mid20b_86

## Result

- Authority: `submission_base_8005.17.zip`
- Authority SHA-256: `c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`
- Targets: 102, 025, 324, 308, 338, 134, 268, 184, 377, 170, 239, 222, 048, 234, 264, 200, 387, 132, 388, 228
- Safe adoptees: **0**
- Score gain: **+0.0**; projected score remains **8005.17**
- Promotion ZIP: not created; protected root files were not changed.

## Exhaustive candidate funnel

- Scanned 12,479 loose ONNX observations and 23,666 task members from 1,259 ZIP files.
- After 4,077 baseline duplicates and 118 oversize observations, 1,140 distinct nonbaseline SHAs remained.
- Fail-closed screen: 319 structure rejects, 58 policy rejects, 665 static-floor rejects, and 98 actual-profile jobs.
- Actual strict-lower candidates: 24. Four-configuration known-perfect: 8. Runtime-shape truthful: 7. Final safe: 0.
- Every actual-lower model was checked on known train/test/arc-gen under disabled/default ORT and 1/4 threads, with runtime/nonfinite/output-shape evidence recorded in `audit/actual_lower_four_config.json`.

| task | base | unique SHA | actual-lower | best cost | known×4 | truthful | decision |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 102 | 493 | 49 | 1 | 491 | 0 | 0 | reject |
| 025 | 474 | 65 | 0 | — | 0 | 0 | reject |
| 324 | 439 | 58 | 0 | — | 0 | 0 | reject |
| 308 | 434 | 58 | 0 | — | 0 | 0 | reject |
| 338 | 426 | 43 | 0 | — | 0 | 0 | reject |
| 134 | 423 | 42 | 1 | 412 | 1 | 0 | reject |
| 268 | 422 | 54 | 0 | — | 0 | 0 | reject |
| 184 | 421 | 43 | 1 | 420 | 0 | 0 | reject |
| 377 | 409 | 63 | 3 | 408 | 0 | 0 | reject |
| 170 | 357 | 46 | 0 | — | 0 | 0 | reject |
| 239 | 384 | 24 | 2 | 374 | 0 | 0 | reject |
| 222 | 380 | 52 | 0 | — | 0 | 0 | reject |
| 048 | 379 | 49 | 13 | 375 | 7 | 7 | reject |
| 234 | 368 | 60 | 0 | — | 0 | 0 | reject |
| 264 | 362 | 64 | 0 | — | 0 | 0 | reject |
| 200 | 346 | 26 | 2 | 344 | 0 | 0 | reject |
| 387 | 337 | 41 | 0 | — | 0 | 0 | reject |
| 132 | 312 | 237 | 0 | — | 0 | 0 | reject |
| 388 | 91 | 32 | 0 | — | 0 | 0 | reject |
| 228 | 302 | 34 | 1 | 294 | 0 | 0 | reject |

## Final blockers

- **task048:** seven cost 378 candidates pass all 270 known examples in all four ORT/thread configurations, strict/data-prop, runtime-shape truth, finite/margin, and Conv-bias UB0. However all score only 457/500 (91.4%) on legal fresh inputs, with the first counterexample at case 11. task048 is in the private-zero catalog; because legal counterexamples exist, no all-input guarantee is possible and the user's private-zero exception does not apply.
- **task134:** cost 423→412 is known-perfect in all four configurations, but direct unsanitized tracing finds six declared/actual shape mismatches (for example `[1,2,1,1]` declared versus `[1,10,30,30]` actual). Rejected as shape cloak.
- **Other 16 actual-lower models:** fail known correctness or an ORT/runtime gate. Notable examples are task102 cost 491 and task377 cost 408 failing optimized ORT, task200 cost 344 scoring 0/84 in all four configurations, and task228 cost 294 failing known.
- Lookup/private-lineage/giant-Einsum/Conv-bias candidates were rejected before fresh scoring. Private/giant artifacts were never admitted from a percentage score alone.

## Exact mechanical coverage

The repository-wide inventory includes retained exact initializer-fusion, outer-product, singleton-axis, dead/no-op, and reuse candidates. The only exact-family strict-lower survivors in this set were the task048 cost-378 variants, which have reproducible legal counterexamples. No initializer-alias/dead-operand artifact in the 20-task set cleared all gates.

## Artifacts

- `inventory/candidate_inventory.json`: target-by-target SHA counts, costs, and funnel summary.
- `rescreen.json`: all 1,140 unique candidates and their stage/reason.
- `audit/actual_lower_four_config.json`: full 24-candidate known×4/static/shape evidence.
- `audit/final_decisions.json`: fail-closed decision record.
- `result.json` and `winner_manifest.json`: zero-promotion result.
