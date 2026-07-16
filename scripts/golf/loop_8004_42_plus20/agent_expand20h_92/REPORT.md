# expand20h_92 — 20-task SHA expansion audit

## Result

No model is approved for fixed adoption. Twelve distinct SHAs are preserved
as `LB_PROBE_REQUIRED` across tasks009, 205, 219, and 396. The best candidate
per task would total `+0.0868517769` if every probe were LB-white;
this is not an accepted or verified gain.

Authority is the exact content retained at `submission.zip` and
`others/71403/lb_verified_8006.61/submission.zip`, both SHA-256
`9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`.
The scan-time alias `submission_base_8006.61.zip` matched that SHA while the
inventory was built but drifted after the scan; it must not be used as the
authority unless restored to the exact SHA.
Fixed champions 013/070/158/254/267/323/379 were not touched. No protected
root artifact or `others` content was modified.

## Inventory and ordinary candidates

All loose ONNX files and matching members of every ZIP were SHA-deduplicated.
The inventory contains 1,374 non-authority SHAs. 163 reached runtime cost
profiling and 43 were runtime-screen strict-lower candidates.

All 43 were independently run on the complete known set under four
configurations: ORT_DISABLE_ALL/default × intra-op threads 1/4, with inter-op
threads fixed at 1. Thirty-eight are known-complete in all four configurations.
After official full-profile cost, shape truth, and policy gates, five clean
candidates reached fresh500:

- task219: three SHAs, each 452/500;
- task365: one SHA, 457/500;
- task009: one SHA, 479/500.

They received two additional independent fresh500 runs. The task009 candidate
remained at or above 95.6%, task365 at or above 90.6%, and the three task219
micro-shaves at or above 87.2% in every ORT mode. The task365 exact SHA was
subsequently matched to a direct 705-pool LB-black report and removed. The
remaining nets are SHA-specific LB probes rather than fixed adoptions because
sampled fresh accuracy is not a proof of LB whiteness.

## Policy reopen

There were 273 initial policy rows. Private/lookup/giant/nonfinite lineage was
treated as net-specific evidence, not a task ban. 105 policy rows were
reopened; 29 had a strict-lower static floor, 25 remained lower in isolated
runtime profiling, and all 25 received the same four-configuration known
audit. Twenty-three are known-complete; after official cost and shape-truth
checks, 14 became `LB_PROBE_REQUIRED`.

Schema-invalid negative Conv padding, strict data-propagation failure, UB,
runtime failure, shape cloak, and actual non-improvement remain hard rejects.

## Exact-SHA history and two-seed classification

The initial set had 19 SHAs. History was applied only when the exact SHA
matched; there is no permanent task blacklist. Five exact historical black or
quarantined nets were removed: task219@1081, task205@937/1010/1015, and
task365@1337. Two
additional task219 lookup nets (@1103 and @1174) were classified
`FALSE_ACCEPT` after both new fresh seeds achieved only 1.2--2.4% at best.

The three task219 micro-shaves (@1445/1453/1454) were intentionally retained
as `LOW`-priority, high-risk LB probes despite 87.2% minimum fresh accuracy:
they are not the exact historical black SHA, and the lane policy keeps such
small shaves for isolated SHA-specific LB measurement. For task396, the exact historical
black@982 and white@1026 SHAs match none of the six current candidates.

## Final probe set

The 12-SHA probe manifest contains:

- task009: 1 SHA, best cost 2616 vs 2619;
- task205: 2 SHAs, best cost 1038 vs 1042;
- task219: 3 SHAs, best cost 1445 vs 1479;
- task396: 6 SHAs, best cost 961 vs 1019.

The complete probe ladder is retained, not just the cheapest net. Safer
fresh-rate fallbacks include task205@1041 and task396@1017; task219 retains
@1453/@1454 as alternative SHAs but all three remain `LOW` priority.

Every probe is strict-lower by official full profiling, complete on known under
all four ORT/thread configurations, runtime-error free in those checks, and
shape-truthful. Lookup/private or imperfect fresh evidence is carried in the
manifest for LB isolation; none appears in `winner_manifest.json`.

## Evidence

- `result.json`: final counts and best-per-task probe projection
- `winner_manifest.json`: empty fixed-adoption manifest
- `probe_manifest.json`: 12 LB probe SHAs with costs, known evidence, and fresh2seed
- `rescreen.json`: all 1,374 inventory rows
- `audit/known_four_all_actual_lower.json`: all 43 ordinary lower candidates ×4
- `audit/policy_reopen.json`: all 273 policy decisions and 25 lower candidates ×4
- `audit/fresh_two_seed.json`: two new fresh500 runs for all initial 19 probes
- `audit/lb_history_exact_sha.json`: exact SHA LB history used for exclusions
- `audit/probe_classification.json`: 12 probe / 5 known-black / 2 false-accept decisions
- `inventory/summary.json`: per-task inventory summary
