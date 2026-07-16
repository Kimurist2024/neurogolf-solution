# expand20j_95 — incremental 20-task scan

## Result

No fixed winner or LB probe candidate survived. Projected gain is `0.0`.
The protected root artifacts and `others/` were not modified.

The only authority used was `submission.zip`, SHA-256
`9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`.
The drifted `submission_base_8006.61.zip` was not used.

## Incremental inventory

The 20 targets were compared against the union of every prior campaign
`rescreen.json`. Seventy-one previously unscanned SHAs remained: task037=1,
task226=22, task014=1, and task036=47. The other 16 targets had no new SHA.

Competition `score_and_verify` profiling left two strict-lower, known-correct
models:

- task014 `15a7de7d...`: cost 370→360 (`+0.027399`), known 266/266 in all four
  ORT/thread configurations, but 17 direct declared/runtime shape
  contradictions. Rejected as shape cloak.
- task037 `df9298f3...`: cost 374→320 (`+0.155935`) and disabled ORT 266/266,
  but both default-ORT configurations fail session construction at a
  `CenterCropPad` axes/shape mismatch. Rejected as runtime-unsafe.

No task226 history was cheaper than the official cost-372 authority.

## task036 cost reconciliation

The truthful runtime profiler initially reported the authority at cost1477,
but this is not the competition comparison. The exact authority member
`574084f0...` contains declared/runtime shape mismatches; competition
`score_and_verify` reports memory255 + params70 = cost325, exactly matching
`all_scores.csv` and prior campaign evidence.
Three independent temporary-directory profiles were identical for both
disputed authorities and all four candidate models; see
`audit/repeat_official_profiles.json`.

The three truthful histories therefore all regress:

- `737fc838...`: cost1051, current fresh 1000/1000.
- `fc83bef4...`: cost1428, current fresh 1000/1000 plus prior 10000/10000.
- `dd794443...`: cost1390, current fresh 999/1000.

Fresh evidence is retained in `audit/fresh_history.json` as diagnostic only.

## task124 repair pivot

After the scan produced no eligible candidate, the lane tried to stabilize the
known one-byte Split-output shave for task124. One- and two-Identity allocator
barriers were inserted before the variadic Split while its unused `r3` output
remained omitted. Both isolated 2,000-case validation processes exited 139
(SIGSEGV), so neither repair is admissible. Evidence is in
`audit/task124_repair_runtime.json`.

## Evidence

- `inventory_delta.json`: prior-inventory SHA delta.
- `audit/official_rescreen.json`: competition profiles and all terminal gates.
- `audit/delta_official_known4.json`: truthful-runtime cost diagnostic,
  known×4, shape and UB evidence.
- `audit/fresh_history.json`: task036 fresh/history diagnostics.
- `audit/repeat_official_profiles.json`: three-run stability check for the
  task014/task036 authority and candidate costs.
- `winner_manifest.json` and `probe_manifest.json`: both empty.
