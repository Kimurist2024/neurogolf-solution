# high142 — task196 / task340 / task354 strict-lower SOUND audit

## Outcome

No candidate is admissible. The winner set is empty and projected gain is
`+0.0`. Root `submission.zip`, score/CSV files, `docs/`, `others/71407/`,
`others/`, and `artifacts/` were not modified.

The immutable authority is LB **8009.46**:

- `submission.zip` SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `submission_base_8009.46.zip` is byte-identical.
- Current members are task196 `94e513b77cfce` (cost 1210), task340
  `d74545851677` (cost 1173), and task354 `4ba066a4ccb0` (cost 536).

The LB-white status of task196 and task354 is bound to those exact member
SHAs. Their malformed shape declarations were not inherited by any descendant.
Every new payload required official-like actual cost strictly below current,
full checker and strict shape/data propagation, standard domains and finite
initializers, Conv/QLinearConv UB0, truthful runtime shapes, known data in four
ORT configurations with raw authority equality, and only then dual-ORT fresh
validation.

## Authority audit

| task | memory + params = cost | full/strict | Conv UB0 | runtime-shape mismatches | known four configs |
|---:|---:|:---:|:---:|---:|:---:|
| 196 | 1049 + 161 = 1210 | pass | pass | 56 | 266/266 all four |
| 340 | 1014 + 159 = 1173 | pass | pass | 0 | 266/266 all four |
| 354 | 461 + 75 = 536 | pass | pass | 7 | 266/266 all four |

The shape trace is decisive: task340 is structurally and dynamically honest;
task196 and task354 are current-only exceptions. Static checker acceptance does
not turn their false declarations into a reusable optimization technique.

## Exact and memory/parameter exploration

The current graphs were scanned for dead nodes and initializers, byte-identical
initializers, identical pure-node CSE, no-ops, optional outputs, constant folds,
safe optimizer passes, shape-preserving rewrites, type-template parameter to
attribute migration, and lower-memory formulations.

- No task contains `ReduceL1` or `ReduceSumSquare`, so the requested
  nonnegative-input replacement by `ReduceSum` is inapplicable.
- Across all three graphs there are no duplicate or unused initializers and no
  exact initializer alias. The only cast-template-only initializer is
  task196's `bool_zero`.
- The conservative optimizer sweep changed task196 and task354 once each but
  did not reduce official-like cost; task340 was byte-unchanged under every
  pass.
- Exact CSE/no-op/value-info/initializer scans produced no strict-lower
  candidate.

### task196

The true rule from `task_810b9b61.py` is: recolor only complete closed blue
rectangle outlines green when both dimensions are at least three; thin or
gapped outlines remain blue. The current compact CNN-like member is not a sound
generator model: the independent dual-ORT 5000-case audit is only 4789/5000 in
each mode, with 211 legal misses, and its truthful declared cost would be
115233.

The independently built packed-bitset true-rule model is proof-backed,
shape-truthful, full/strict clean, known-correct, and recorded fresh 3000/3000.
Its cost is 5573 (memory 5368 + params 205), which is 4363 above current. The
only remaining local type migration, `CastLike(g_raw,bool_zero)` to `Cast`, is
raw-equal on all 266 known examples in both runtimes but exposes a larger real
intermediate and costs 1433, 223 worse. The `Greater(g_raw,0)` variant also has
an asymmetric ORT allocator failure.

Five retained archive SHAs were reprofiled at actual costs 1495, 1518, 1773,
1773, and 1774. None is lower than 1210. Prior exact anchor/flood models and
sampled compact models are either more expensive or retain generator
counterexamples.

### task340

The true rule from `task_d687bc17.py` assigns top/right/bottom/left roles from
the four outer-border colors, moves same-colored interior markers to the
corresponding inner boundary lane, and clears unrelated markers.

The current 37-node model is already honest: zero `value_info`, zero runtime
shape mismatch, no dead/unused/duplicate structure, and dual-ORT fresh
5000/5000 in both modes with zero generation or runtime errors. This exact SHA
is the same payload independently audited in lane B19.

All exact reuse families are exhausted: initializer and model-wide reuse,
Einsum factor reuse, low-rank/proportional/slicing/tensor-mode reuse,
shared-operand fusion, inlining, signed scale/permutation absorption, CSE, and
dead/no-op shaving produced no hit. `ACTable[6,9,1]` has 54 parameters and exact
real rank 5; dense factorization requires `5*(6+9)=75` parameters before
materialization, so it cannot lower cost. Eighty-four historical rows bottom
out at 1173 and 48 prior optimizer variants found nothing cheaper. The retained
archive scan had no non-authority candidate.

### task354

The true rule from `task_ddf7fa4f.py` uses the three colored lights in the top
row to recolor each of the three non-overlapping gray rectangles aligned below
them. The current exact SHA is the LB-accepted cost-536 payload and has recorded
fresh results of 500/500 on two independent seeds in both ORT modes, but its
seven false runtime shape declarations make that exemption SHA-specific.

Three explicit parameter-to-attribute probes were measured:

| rewrite | memory + params = cost | runtime mismatches | decision |
|:---|---:|---:|:---|
| `shape12_dyn` CastLike -> Cast(INT64) | 461 + 75 = 536 | 7 | not lower |
| `idx_i32` CastLike -> Cast(INT32) | 937 + 75 = 1012 | 7 | worse |
| both rewrites | 937 + 75 = 1012 | 7 | worse |

The type template is still used elsewhere, so the first rewrite does not save
a parameter. The second exposes the true `[1,1,12,10]` tensor and adds 476
memory. The sole retained archive model profiles at cost 560, above 536; prior
harvested alternatives bottom out at the same actual cost.

## Gate result

No new payload cleared the first combined gate of official strict-lower cost,
full/strict structure, UB0, and truthful runtime shapes. Consequently, known
raw-equivalence and fresh validation were not run for a new candidate; neither
can change an already non-lower or shape-invalid admission result. There is
nothing to merge or promote.

Primary evidence:

- `authority_audit.json`: authority hashes/profiles, structure, shape traces,
  and four known ORT configurations.
- `exact_scan.json`, `initializer_analysis.json`, and
  `optimizer_sweep.json`: exact local and parameter/memory scans.
- `targeted_rewrite_audit.json`: Reduction applicability and task354 attribute
  migration measurements.
- `archive_rescreen.json`: retained archive revalidation.
- `manifest.json` and `winner_manifest.json`: machine-readable disposition.

