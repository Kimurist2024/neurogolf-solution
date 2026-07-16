# A41 — task366 strict exact-golf result

No candidate is eligible for promotion. The authoritative task366 member in
`submission_base_8002.63.zip` remains unchanged at cost **7987**
(`memory=7622`, `params=365`, score `16.01442950108261`). The ZIP SHA-256 is
`a2da30657f3798e861f369ac896f36722ff658ed3e468c4d55db9a04eefbccfc` and
the task366 member SHA-256 is
`072ca4f43fb6fe6e96cf90826d5f9b2dbdbe5016f7db3daa14f265620c1a010a`.

## What was exhausted

The exact structural scan found one `Identity`, zero same-type `Cast`, zero
same-type `CastLike`, zero exact structural CSE pairs, zero duplicate
initializers, zero unused initializers, and zero dead nodes. Removing the sole
Identity is raw-exact on all 255 executable known cases in both ORT modes, but
the annotation-free probe is not scorable. Repeating that rewrite while
retaining repaired shapes gives the clean exact control:

| model | memory | params | cost | result |
|---|---:|---:|---:|---|
| authority | 7622 | 365 | 7987 | current, shape-cloaked |
| truthful annotations | 9186 | 365 | 9551 | exact, too expensive |
| truthful + Identity bypass | 9178 | 365 | 9543 | exact, too expensive |

The clean rewrite saves exactly 8 memory units, but remains 1556 cost above
the authority. Even hypothetically deleting all 365 initializer elements while
leaving truthful intermediate memory unchanged would still cost 9178, which is
1191 above the authority. Initializer-only factorization therefore cannot
cross the current bound; a large semantic graph rewrite would be required.

## Shape-cloak audit

On one real task366 training input, the authority has 98 declared-versus-actual
intermediate shape mismatches and 22 `CenterCropPad` nodes. Repairing all 98
declarations makes strict ONNX checker/inference pass and raises measured memory
from 7622 to 9186. Thus the current 7987 figure is not compatible with the
requested truthful-shape gate.

Historical cheaper candidates inherit the same defect:

| candidate cost | shape mismatches | random differential | verdict |
|---:|---:|---|---|
| 7985 | 100 | 477/500 raw-equal; 23 mismatches | reject |
| 7916 | 92 | 477/500 raw-equal; 23 mismatches | reject |
| 7646 | 107 | 2894/2998 raw-equal; 104 mismatches | reject |

The external validator was run with `--allow-random-mismatch` only to preserve
diagnostics. Its permissive label is not a promotion verdict; the measured raw
and threshold mismatches violate the lane's exact-behavior requirement.

## Validation and artifacts

The authority, exact controls, and historical models all complete 255 known
cases with zero runtime errors in both `ORT_DISABLE_ALL` and default ORT modes;
11 oversized examples are skipped exactly as in the official local path. This
also demonstrates why known-only validation cannot approve the historical
leads—their random behavior differs.

Evidence is in `control_costs.json`, `known_dual_all.json`,
`runtime_shape_trace.json`, `historical_shape_audit.json`, and
`historical_random500_summary.json`. No root ZIP, CSV, or submission artifact
was modified by this lane. The pre-existing root `all_scores.csv` worktree
change was left untouched.
