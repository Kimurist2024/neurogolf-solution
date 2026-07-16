# new-mid22 8-task audit — latest 8005.16 baseline

## Result

Eight newly assigned files were independently extracted from
`submission_base_8005.16.zip` (SHA-256
`73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`)
and audited. **No model is eligible for adoption; projected gain is +0.0.**
No ZIP, score CSV, best-score pointer, or shared handcrafted artifact was
modified.

| task | cost | dual known | dual fresh (500) | decisive result |
|---:|---:|---:|---:|---|
| 123 | 266 | 265/265 | 500/500 | sound floor; no exact micro-shave exists |
| 316 | 246 | 266/266 | 500/500 | only repeated-axis probe fails at the first known case; compact graph uses ScatterElements |
| 212 | 240 | 265/265 | 500/500 | 48-input giant Einsum; truthful control costs 4398 |
| 301 | 240 | 266/266 | 500/500 | 51-input giant Einsum; truthful control costs 1991 |
| 055 | 234 | 263/263 | 482/493 compatible | 11 fresh errors in each ORT mode and 28-input giant Einsum; truthful control costs 9326 |
| 086 | 221 | 266/266 | 500/500 | 80 CenterCropPad nodes and false output shape; truthful control costs 17097 |
| 163 | 196 | 267/267 | 500/500 | 53-input giant Einsum; two cost-184 prunes are 0/267 known |
| 206 | 194 | 266/266 | 500/500 | CenterCropPad/ScatterElements and three false runtime shapes; truthful control costs 7753 |

Fresh results are identical under `ORT_DISABLE_ALL` and default ORT. For
task055, seven of 500 generated grids exceed the 30x30 model contract and are
not convertible; the rate over compatible inputs is 482/493 = 97.7688%, with
11 true output mismatches.

## Newly tested probes

- task316 `collapse_ones10_axis2`, SHA
  `0af6539d7c56953cedba92a5b2c303a531b56b2bb0e00ba24390529bffdcc7f2`:
  nominal initializer count falls by nine, but both ORT modes fail immediately
  because ScatterElements indices and updates have dimensions 19 and 10.
- task163 root-sweep prune r001, SHA
  `13ac6f3231a9b59efc2173c79479f40bc90f88409cae13aed87b875cedd92253`:
  cost 184, known 0/267 in both modes, runtime errors 0, reject.
- task163 root-sweep prune r002, SHA
  `6bae6b57bbccd4dcd4a3a90767fcd50fd62ba170c0daf0ba49a2b3239a69c8dd`:
  cost 184, known 0/267 in both modes, runtime errors 0, reject.

The task163 prunes were not sent to fresh testing because complete known
correctness is a mandatory prerequisite. Both also retain the prohibited
53-input giant Einsum.

## Floor evidence

The exact micro-search checked repeated initializer axes, exact constant
aliases, optional zero inputs, duplicate expressions, and identity removal.
Only the task316 probe above was constructible, and it failed. Existing deeper
search evidence was also reconciled with the exact 8005.16 members:

- task123: all six CP component deletion probes fail; the current max-Einsum-6
  graph is strict, truthful, UB-free, and fresh-exact.
- task212: 123 single-axis collapses and six coupled rank reductions previously
  failed; the incumbent has zero intermediate memory.
- task301: 22 unique repository models contain no sub-240 candidate; exact
  pair contractions and initializer relations do not save parameters.
- task055: all archived sub-baseline profiles retain giant Einsums; the fresh
  failures independently show that cost alone is insufficient.
- task086: the repository screen covered 57 unique alternatives with no
  truthful cheaper winner.
- task163: the known factor/deletion family at costs 136, 184, and 188 fails
  known correctness.
- task206: the latest cost-194 graph already removed two Shrink intermediates
  from the cost-196 lineage; a truthful specification control remains far over
  the baseline.

Machine-readable evidence is in `baseline_audit.json`,
`baseline_fresh_dual500.json`, `sound_controls_audit.json`,
`micro_search.json`, `task163_root_prunes_audit.json`, and `result.json`.
