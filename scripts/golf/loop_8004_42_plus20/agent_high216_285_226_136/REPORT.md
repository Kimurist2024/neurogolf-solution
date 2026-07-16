# high136 — task216 / task285 / task226 strict-lower SOUND audit

## Outcome

No candidate is admissible. The winner set is empty and projected gain is
`+0.0`. Root `submission.zip`, `all_scores.csv`, `others/`, and `artifacts/`
were not modified.

The immutable authority is LB **8009.46**:

- `submission.zip` SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `submission_base_8009.46.zip` has the same SHA-256.
- Current members are task216 `9a5f4f10d6e0` (cost 1499), task285
  `366212e29105` (cost 8623), and task226 `852b6091385d` (cost 372).

The task216/task285 current members retain LB-white status only as those exact
SHAs. Their malformed shape declarations were not transferred as an exemption
to descendants. Every new payload required full/strict structure, truthful
runtime shapes, official-like actual cost, Conv UB0, known data in both ORT
modes at 1/4 threads, authority raw equivalence, and then fresh two-seed
validation at at least 1500 cases per seed and 90% per seed.

## Authority audit

| task | official memory + params = cost | full/strict | Conv UB0 | runtime-shape mismatches | known four configs |
|---:|---:|:---:|:---:|---:|:---:|
| 216 | 1453 + 46 = 1499 | pass | pass | 53 | 266/266 all four |
| 285 | 8322 + 301 = 8623 | pass | pass | 57 | DISABLE_ALL 265/265; default session fail |
| 226 | 333 + 39 = 372 | pass | pass | 0 | 133/133 all four |

The task285 default failure is an existing `Concat` inferred/declared shape
conflict. The baseline defects are evidence, not permission to admit a new
shape cloak.

## Mechanical exact exploration

The graphs were scanned for dead nodes/initializers/value-info, initializer
aliases, identical pure-node CSE, no-ops, constant folds, optional outputs,
shape-preserving Boolean/integer identities, and initializer-to-attribute
migration.

### task216

- Bypassing `Identity(sl18_i32 -> sl20_dyn)` exposes the real Slice length and
  fails strict inference at `rowu` and `CenterCropPad(prev)`.
- Four exact constant folds (`shape4_dyn`, `sl20_dyn`, `rowu_len_minus`, and
  `rowu_len_plus`) expose the real 4/16/17/18-length intermediates and fail
  full/strict inference.
- Clearing value-info removes the static-positive proof for `roiout` and the
  pad-shape path, so it is not admissible.
- No initializer aliases, unused initializers, type-only `CastLike`
  initializer, or safe parameter-to-attribute migration exists.

The optimizer's only apparently lower payload was
`task216_eliminate_deadend_5c0e9ddf5f49.onnx`:

| metric | current | optimizer payload |
|---|---:|---:|
| declared memory + params | 1453 + 46 | 979 + 46 |
| declared cost | 1499 | 1025 |
| official-like actual cost | 1499 | **1499** |
| runtime-shape mismatches | 53 | 53 |
| known four configs | 266/266 | 266/266, raw-equal |

Its nodes, initializers, graph inputs/outputs, and opsets are unchanged; only
`graph.value_info` differs. The competition profiler restores the actual cost,
so this is a value-info/shape-cloak mirage and is rejected before fresh.

The independently validated truthful true-rule control costs 31511, has zero
shape mismatches, passes known 266/266 in both modes and fresh 2 x 5000 in both
modes. It is +30012 over current and cannot be a replacement. The current
compact member itself has 1667 runtime errors in each 5000-case placement seed,
which confirms that its LB exemption cannot be generalized.

### task285

- Removing or folding `Identity(__zd_shape_const_zero300 ->
  __zd_shape_zero300)` reveals the true length-300 `CenterCropPad(zero300)`
  output against the declared length 1 and fails strict inference.
- Combining that rewrite with the no-op `Reshape(cm_post_0 <- biton)` also
  exposes a 2-versus-1 shape conflict.
- Clearing value-info leaves nonstatic shape tensors and is inadmissible.
- No initializer aliases, unused initializers, type-only `CastLike`
  initializer, or safe parameter-to-attribute migration exists.

The minimum known fixture-lookup-free, runtime-shape-truthful true-rule model
costs 14685 (13016 memory + 1669 params), passes known 265/265 in both modes,
and is +6062 above current. The best older compact archive is cost 8717 and is
also not lower than 8623.

### task226

The current graph already is the accepted six-position Boolean rule model from
`agent_target_mid19`, byte-for-byte:

- full/strict structure and Conv UB0 pass;
- zero runtime-shape mismatches in both ORT modes;
- known 133/133 in all four configurations;
- complete generator domain 136/136 in both modes, raw-equal to the pinned
  baseline;
- fresh seeds 22650001 and 22650002: 5000/5000 in both modes.

It uses the irreducible six input-column positions `{1,2,3,5,6,8}` followed by
the Boolean `Where` DAG and one `QLinearConv`. No identical initializer, unused
initializer, duplicate pure node, type-only `CastLike` initializer, or
mechanical exact optimizer change exists. The best older archive costs 375,
above current 372.

## Archive revalidation

| task | unique non-authority SHAs | minimum declared cost | minimum official actual cost where needed | strict-lower survivors |
|---:|---:|---:|---:|---:|
| 216 | 19 | 1037 | 1511 | 0 |
| 285 | 13 | 8717 | not run: declared already >= 8623 | 0 |
| 226 | 12 | 375 | not run: declared already >= 372 | 0 |

For task216, four archives declared at 1037 all profile at actual cost 1511;
the remaining actual-profiled candidates are at least 1534. There is no archive
payload that reaches the structural/runtime/fresh stages.

## Gate result

No new payload cleared official strict-lower cost plus truthful runtime shapes.
Fresh was therefore not run for a new candidate; doing so could not change any
admission decision. There is nothing to merge or promote.

Evidence:

- `authority_audit.json`: authority SHAs, official profiles, structural gates,
  runtime-shape traces, and known-four execution.
- `exact_scan.json`: mechanical exact transforms and structural rejections.
- `initializer_analysis.json`: alias/use/parameter audit.
- `optimizer_sweep.json`: conservative optimizer-pass sweep.
- `optimizer_candidate_audit.json`: deep rejection of the task216 value-info
  candidate.
- `archive_rescreen.json`: deduplicated archive revalidation.
- `manifest.json` and `winner_manifest.json`: machine-readable disposition.
