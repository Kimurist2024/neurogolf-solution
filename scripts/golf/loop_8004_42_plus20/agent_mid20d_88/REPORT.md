# Expanded mid-cost 20-task audit — mid20d_88

## Result

- Authority: `submission_base_8005.17.zip`
- Authority SHA-256: `c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`
- Targets: 055, 031, 086, 088, 042, 143, 247, 079, 065, 344, 115, 163, 206, 114, 273, 161, 071, 105, 259, 189
- Safe adoptees: **0**
- Score gain: **+0.0**; projected score remains **8005.17**
- Promotion ZIP: not created; protected root files were not changed.

## Candidate funnel

- Scanned 12,030 loose ONNX observations and 23,671 members from 1,259 ZIP files.
- After baseline duplicate/oversize filtering, 803 distinct nonbaseline SHAs remained.
- Fail-closed stages: 176 structure rejects, 6 policy rejects, 567 static-floor rejects, and 54 actual-profile jobs.
- Actual strict-lower: 28; known-perfect in disabled/default ORT × 1/4 threads: 22; runtime-shape truthful: 0; safe: 0.
- Fresh two-seed testing was not started because no model passed the earlier known×4 and truthful-shape gates.

| task | base | unique SHA | actual-lower | best cost | known×4 | truthful | decision |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 055 | 234 | 50 | 0 | — | 0 | 0 | reject |
| 031 | 183 | 47 | 0 | — | 0 | 0 | reject |
| 086 | 221 | 75 | 0 | — | 0 | 0 | reject |
| 088 | 902 | 48 | 26 | 134 | 22 | 0 | reject |
| 042 | 217 | 33 | 0 | — | 0 | 0 | reject |
| 143 | 212 | 42 | 0 | — | 0 | 0 | reject |
| 247 | 212 | 23 | 0 | — | 0 | 0 | reject |
| 079 | 209 | 46 | 0 | — | 0 | 0 | reject |
| 065 | 199 | 40 | 0 | — | 0 | 0 | reject |
| 344 | 197 | 25 | 0 | — | 0 | 0 | reject |
| 115 | 197 | 27 | 0 | — | 0 | 0 | reject |
| 163 | 196 | 35 | 0 | — | 0 | 0 | reject |
| 206 | 194 | 56 | 0 | — | 0 | 0 | reject |
| 114 | 194 | 21 | 0 | — | 0 | 0 | reject |
| 273 | 193 | 34 | 0 | — | 0 | 0 | reject |
| 161 | 190 | 39 | 1 | 186 | 0 | 0 | reject |
| 071 | 188 | 52 | 1 | 186 | 0 | 0 | reject |
| 105 | 188 | 40 | 0 | — | 0 | 0 | reject |
| 259 | 187 | 35 | 0 | — | 0 | 0 | reject |
| 189 | 183 | 35 | 0 | — | 0 | 0 | reject |

## Final blockers

- **task088:** 26 actual-lower candidates were found. Twenty-two pass all 267 known examples in all four ORT/thread configurations, but 21 have 11–18 direct declared/actual runtime-shape mismatches. The remaining cost-211 graph cannot be truthfully traced because it contains duplicate node name `label_scale`. All are rejected as shape-cloak/schema-invalid artifacts. The cheapest task088 leads that are not known-perfect score 0/267, 23/267, or fail optimized ORT.
- **task071:** cost 188→186 misses 1 of 265 known cases in every configuration. The retained exact CastLike rewrite is also covered by the repository scan and was previously rejected for giant-Einsum/default-vs-disable instability.
- **task161:** cost 190→186 misses 1 of 266 known cases in every configuration.
- The other 17 targets have no policy-clean actual-lower survivor. task086 is private-catalog lineage; no percentage-only artifact was admitted. Lookup, giant-Einsum, Conv-bias UB, schema-invalid, and private-lineage artifacts were fail-closed before fresh evaluation.

## Exact initializer/dead/no-op coverage

All 20 authority members are byte-identical between 8005.16 and 8005.17, so the prior full-400 exact initializer/Einsum and dead/no-op sweeps remain applicable. The repository-wide inventory also directly included their emitted artifacts, including task163 latent prunes and task071 CastLike. None cleared current known×4, truthful-shape, and no-giant gates. See `audit/baseline_8005_16_to_8005_17_equivalence.json`.

## Artifacts

- `inventory/candidate_inventory.json`: task-level SHA inventory and funnel.
- `rescreen.json`: all 803 unique candidate decisions.
- `audit/actual_lower_four_config.json`: 28-candidate known×4/static/runtime-shape evidence.
- `audit/final_decisions.json`: final fail-closed classifications.
- `result.json` and `winner_manifest.json`: zero-promotion result.
