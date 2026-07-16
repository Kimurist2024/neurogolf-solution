# Expanded 20-task audit — agent_expand20f_90

## Result

- Immutable authority: `submission_base_8006.61.zip`
- Authority SHA-256: `9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`
- Safe fixed adoptees: **0**
- Final `LB_PROBE_REQUIRED`: **0**
- Score gain: **+0.0**; projected score remains **8006.61**
- Locked champions 013/070/158/254/267/323/379 were untouched. No root/others/ZIP mutation was performed.

## Exhaustive funnel

- Scanned 12,283 loose observations and 23,712 task members from 1,261 ZIP files; SHA dedup left 1,025 nonbaseline candidates.
- Ordinary fail-closed path profiled 88 models. A second pass deliberately reopened 113 giant/lookup/private models rather than rejecting those labels alone.
- Combined actual strict-lower: 49; known-complete in disable/default ORT × 1/4 threads: 13; truthful runtime shapes: 6.
- The six truthful models were all task185 lookup/private networks. SHA-specific LB evidence and two fresh-500 seeds reduce them to three known LB-black plus three local false accepts. Final probe and fixed-safe sets are empty.

| task | base | unique SHA | actual-lower | best cost | known×4 | truthful | probe |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 075 | 345 | 28 | 0 | — | 0 | 0 | 0 |
| 392 | 345 | 39 | 5 | 341 | 0 | 0 | 0 |
| 225 | 333 | 33 | 1 | 306 | 0 | 0 | 0 |
| 218 | 329 | 38 | 1 | 314 | 0 | 0 | 0 |
| 159 | 293 | 42 | 3 | 236 | 0 | 0 | 0 |
| 185 | 279 | 48 | 8 | 185 | 6 | 6 | 0 |
| 263 | 181 | 35 | 0 | — | 0 | 0 | 0 |
| 370 | 954 | 70 | 0 | — | 0 | 0 | 0 |
| 182 | 951 | 54 | 0 | — | 0 | 0 | 0 |
| 330 | 896 | 54 | 3 | 807 | 0 | 0 | 0 |
| 361 | 858 | 66 | 3 | 810 | 0 | 0 | 0 |
| 157 | 853 | 77 | 0 | — | 0 | 0 | 0 |
| 280 | 828 | 65 | 1 | 827 | 0 | 0 | 0 |
| 382 | 820 | 62 | 14 | 762 | 0 | 0 | 0 |
| 201 | 789 | 84 | 3 | 543 | 3 | 0 | 0 |
| 251 | 755 | 42 | 2 | 582 | 0 | 0 | 0 |
| 012 | 710 | 16 | 1 | 500 | 0 | 0 | 0 |
| 107 | 708 | 77 | 2 | 638 | 2 | 0 | 0 |
| 131 | 691 | 50 | 2 | 596 | 2 | 0 | 0 |
| 364 | 685 | 45 | 0 | — | 0 | 0 | 0 |

## task185 SHA-level disposition

- `819bc2f00d96` cost 185: **KNOWN_LB_BLACK**, fresh 0.2%/0.2%. 70207 third task185 cost185 was directly LB-black.
- `ce35307db278` cost 186: **REJECT_LOCAL_FALSE_ACCEPT**, fresh 0.2%/0.2%. same lookup family; independent fresh seeds both 1/500 (0.2%).
- `d3da20db8d73` cost 186: **KNOWN_LB_BLACK**, fresh 0.2%/0.2%. 70203 task185_cost186_verified_lowcost lineage was the 70203 culprit.
- `d675c9b4b81c` cost 190: **REJECT_LOCAL_FALSE_ACCEPT**, fresh 0.2%/0.2%. same lookup family; independent fresh seeds both 1/500 (0.2%).
- `e086e1c96c9f` cost 191: **REJECT_LOCAL_FALSE_ACCEPT**, fresh 0.2%/0.2%. same lookup family; independent fresh seeds both 1/500 (0.2%).
- `d21f1db4d69b` cost 273: **KNOWN_LB_BLACK**, fresh 100.0%/100.0%. cost273 task185 network was LB-black in the later partner probe.

The cost-273 `d21f1db...` model is especially important: it scores 500/500 on both fresh seeds yet is already LB-black. Fresh accuracy is therefore retained as a ranking signal, not treated as a white guarantee.

## Other lower leads

- task107 cost 706/638, task131 cost 627/596, and task201 cost 785/682/543 are known×4 complete but fail direct truthful runtime-shape tracing; they are hard rejects, not probes.
- task251 cost 709/582 is 266/266 under disabled ORT but default ORT fails all cases during session construction. The task-specific recent black warning is therefore moot for these SHAs: they fail the local runtime gate first.
- task382 contributes fourteen lower models across both paths; they miss known cases or fail default ORT. task012 is 235/265, task159 lower giant models fail known, and remaining lower leads likewise fail known/runtime.
- Conv-bias UB, schema-invalid, noncanonical-I/O, and actual non-improvements remain hard rejects. Giant/lookup/private labels alone were not used as the final rejection reason.

## Authority and exact/no-op coverage

All 20 target members are byte-identical between 8005.17 and 8006.61; see `audit/authority_member_identity.json`. Therefore the prior all-400 initializer reuse/dead/no-op results remain applicable. Their emitted candidates, including task107 initializer reuse and task382 truthful repair, were also included in the repository-wide scan. No all-input equivalence proof produced a strict-lower safe winner.

## Artifacts

- `inventory/candidate_inventory.json`
- `rescreen.json`
- `audit/actual_lower_four_config.json`
- `audit/reopened_giant_lookup_private.json`
- `audit/fresh_probe_2seed.json`
- `audit/lb_history_classification.json`
- `probe_manifest.json`, `winner_manifest.json`, and `result.json`
