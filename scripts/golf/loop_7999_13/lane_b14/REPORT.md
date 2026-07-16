# Lane B14 report — task005/task080 exact-base audit

Base: `submission_base_7999.13.zip` (task005 cost 2325; task080 cost 3051).
No project-level ZIP, CSV, or score ledger was modified by this lane.

## Result

- Winners: **0**; aggregate score gain: **+0.00**.
- The deduplicated scan covered 113 task005 models / 970 raw sources and 77
  task080 models / 929 raw sources. It found no strict-cost candidate.
- task005's lowest nonbase static floor is 2325 (a tie); the next floor is
  2329. task080's lowest nonbase floor is 3053, already above its 3051 base.
- All exact baselines pass checker, strict data-propagating shape inference,
  static-positive shapes, standard domains, banned-op/nested/function/sparse/
  external/nonfinite/giant-Einsum checks, and Conv bias safety.
- All 48 installed optimizer passes were probed individually plus the default
  full sweep. No valid strictly cheaper model was produced.

## Dual-ORT validation

- Known: task005 266/266 and task080 231/231 in both ORT modes; errors 0.
- Fresh seed 140799913: task005 4967/5000 = 99.34% in both modes; errors 0.
  The 33 semantic misses are duplicate-guide-color selector failures, not
  runtime/session errors. This meets the lane's explicit >=95% base-equivalent
  allowance, but is not generator-sound.
- Fresh seed 140799913: task080 5000/5000 in both modes; errors 0 (1001 >30x30
  generated cases skipped per the scorer contract in each stream).
- Minimum observed nonzero absolute output margin is 1.0 throughout.

## Promotion decision

Nothing is promoted. The sound task005 rebuild costs 2389 (+64), and the exact
task080 model is already below every stored alternative. `winner_manifest.json`
therefore stays empty and records runtime errors = 0.
