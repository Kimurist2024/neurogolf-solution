# Private/lookup lower-candidate bundle audit

## Outcome

No candidate is admissible. Winner count is **0** and projected gain is
`+0.0`. The authority is `submission_base_8005.17.zip`, SHA-256
`c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`.
The 8005.16 to 8005.17 update changes task226 only; all six audited members
are byte-identical across the two ZIPs.

No root ZIP, protected score file, CSV, canonical optimized model, or score
ledger was modified.

## Candidate-file coverage

The audit selected the larger byte-distinct retained frontier for each task
from `lane_archive_loose_sweep` and `lane_archive_all400`, then added five later
strict-lower constructions: the task219 B32 algebraic reduction and four
task396 true-rule attempts. Files were deduplicated by `(task, SHA-256)` and
reprofiled with official-like runtime memory.

| task | baseline | strict-lower files | known dual perfect | lookup/scatter | cloak | giant Einsum |
|---:|---:|---:|---:|---:|---:|---:|
| 134 | 423 | 8 | 6 | 7 | 4 | 0 |
| 202 | 48 | 3 | 3 | 0 | 0 | 3 |
| 219 | 1479 | 6 | 5 | 6 | 1 | 0 |
| 271 | 135 | 2 | 1 | 1 | 0 | 0 |
| 343 | 173 | 2 | 2 | 0 | 0 | 0 |
| 396 | 1019 | 12 | 9 | 12 | 4 | 0 |
| **total** | — | **33** | **26** | — | — | — |

Every one of the 33 files is strictly lower by actual official-like profiling,
not just its archived static declaration. Complete per-file paths, hashes,
costs, known dual results, operation inventories, strict/data-propagation
results, and runtime-shape traces are in `extended_candidate_audit.json`.

## Generator rules and decisions

### task134 / `5ad4f10b`

The input contains sparse noise in one color and a magnified 3x3 Conway sprite
in another. The output is the sprite occupancy bitmap recolored with the noise
color. A correct compiler must infer color role, scale, bbox, and nine samples.

All eight lower files fail known data, contain lookup machinery, or carry
runtime-shape cloaks. The strongest truthful lookup families are not complete:
r04 scores 4840/5000 disabled and 4823/5000 default; r06 scores 4803/5000 and
4825/5000. Verified true-rule controls exist, but cost 1555 and 3411 versus the
423 baseline. Decision: **reject all eight**.

### task202 / `855e0971`

Each horizontal or transposed color stratum contains zero to two holes; a hole
erases its complete column/row within that stratum. The three cost20/28 files
are one-node 16-25-input numerical contractions, not complete compilers.

All pass the 230 convertible known cases in both modes, but r01 fails 10/422
fresh and r02 fails 9/422. r03 has a generator-valid counterexample reproduced
with the same wrong 20 cells under disabled/default and threads 1/4. The clean
spec-derived construction costs 2472, above 48. Decision: **reject all three**.

### task219 / `90f3ed37`

Rows contain repeated A, B, and C cyan patterns; the first group reveals C and
later groups must receive blue copies of C. Historical cost1081-1467 files are
lookup/private approximations. Their known-pass fresh results are only 32/500,
9/500, 7/500, and 417/500.

The later cost1445 B32 file exactly preserves incumbent raw behavior through
algebraic simplifications, but it is not a true-rule reconstruction and scores
4327/5000 fresh. It also retains ScatterElements. Decision: **reject all six**.

### task271 / `ae4f1146`

Four non-overlapping 3x3 cyan boxes have strictly increasing blue-pixel counts;
the output is the last/largest pattern. The cost10 file is 0/267 known. The
cost126 file is dual 267/267 but contains nine TfIdfVectorizer nodes and scores
only 2/5000 generated cases in each optimizer. Decision: **reject both**.

### task343 / `d8c310e9`

The task extends a visible periodic 3- or 4-column pattern across a 5x15 grid,
with optional alternating reflection. Both cost172 candidates are structurally
clean and known 266/266, but they are compact feature classifiers rather than
the exact period rule. Independent fresh streams yield roughly 4965-4976/5000,
with deterministic counterexamples. The exact lookup-free period compiler
costs 178, five above the 173 baseline. Decision: **reject both**.

### task396 / `fcb5c309`

The uniquely widest/tallest hollow rectangle must be cropped, with its border
and retained static pixels recolored using the other nonzero color. Random
static and other rectangles defeat global frequency and short run heuristics.

All twelve lower files contain Hardmax/Scatter-style lookup machinery; four
also have runtime-shape failures/cloaks. Known-perfect archive costs947-965
score only 4861-4908/5000 fresh. The cost1014 occupancy attempt reaches
4963/5000, still with 37 deterministic failures. The verified generator-SOUND
corner parser costs 1245, above 1019. Decision: **reject all twelve**.

## Gate conclusion

Known-dual agreement, or even fresh accuracy above 90%, was not treated as a
private guarantee. No lower file has complete finite-support/equivalence-class
proof, and no lookup-free, shape-truthful true-rule compiler is below its
authority member. Therefore `winner_manifest.json` is intentionally empty.

Primary machine evidence:

- `authority_members.json`: 8005.17 hashes and member identity
- `retained_audit.json`: initial 22-file official profiling and known audit
- `extended_candidate_audit.json`: full 33-file deduplicated frontier
- `result.json`: per-task disposition summary
