# task013 finite-support expansion audit

## Outcome

Eight strict-lower task013 files were audited against immutable
`submission_base_8005.17.zip` (`c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`).  Exactly one
non-duplicate winner is selected: `task013_r001.onnx`, SHA-256 `ad4eb35978f3e38d1d3e2afdd55e55db871962cc2ea4c989675d9d583434103b`.
Its actual cost is `638 -> 636`, for projected gain `0.003139720005`.  No ZIP,
`all_scores.csv`, or protected root file was modified.

## Full reachable-support guarantee

The generator has 37,800 structural states.  The selected giant-Einsum model
executed every state independently in all four required ORT configurations:

| configuration | correct | wrong | runtime errors | nonfinite | (0,0.25) positives | min positive |
|---|---:|---:|---:|---:|---:|---:|
| disable_all_threads1 | 37800/37800 | 0 | 0 | 0 | 0 | 4.6757812 |
| disable_all_threads4 | 37800/37800 | 0 | 0 | 0 | 0 | 4.6757812 |
| default_threads1 | 37800/37800 | 0 | 0 | 0 | 0 | 4.6757812 |
| default_threads4 | 37800/37800 | 0 | 0 | 0 | 0 | 4.6757812 |

This is 151,200 platform executions.  The 72 ordered distinct nonzero colour
pairs are rigorously reduced, not sampled: geometry uses `nz_f` and is colour-ID
independent; marker colours are exactly recovered by
`S=c0+c1`, `T=c0*p0+c1*(p0+d)`; the terminal colour score is
`0.25-(k-c)^2`, positive iff `k=c`; the background score is
`0.0625-1000000*k^2`, positive iff `k=0`.  All 48,600 reachable float16 colour
recovery combinations and all selector values were mechanically checked.
Thus the 37,800 executions cover all 2,721,600 generator parameter states.

## Rewrite and static gates

The candidate removes `T_zero=[1,0]` and replaces its six operand uses across
four Einsum nodes by the exact `Qor` main diagonal.  That diagonal is exactly
`[1,0]`; all other initializers are byte-identical, and 51/55 nodes are
byte-identical.  All 55 runtime node-output shapes match strict inferred shapes,
with zero nonfinite intermediate outputs.

Independent gates pass: actual profiler `488 memory + 148 params = 636`, ONNX
full checker, strict shape inference with data propagation, positive static
shapes, standard domains, Conv-family UB0, lookup0, nested graph/function/sparse0,
banned-op0, finite initializers, and known `267/267` in each of the four modes.

## Expanded candidate-file set

| # | file | SHA | cost | static/proof | known x4 | decision |
|---:|---|---|---:|---|---|---|
| 1 | `task013_r001.onnx` | `ad4eb35978f3…` | 636 | yes | yes | selected |
| 2 | `task013_r002.onnx` | `a8577f8053f6…` | 636 | yes | yes | alternate |
| 3 | `task013_r003.onnx` | `379452bc9d19…` | 636 | yes | yes | alternate |
| 4 | `task013_r004.onnx` | `b22924f95f7d…` | 636 | yes | yes | alternate |
| 5 | `task013_r005.onnx` | `dc1d4ccb3722…` | 636 | yes | yes | alternate |
| 6 | `task013_r006.onnx` | `5fb80b2bd47a…` | 636 | yes | yes | alternate |
| 7 | `task013_r009.onnx` | `e99aaf228a00…` | 636 | yes | yes | alternate |
| 8 | `task013_r010.onnx` | `3ed3d20fb0c5…` | 636 | yes | yes | alternate |

All seven alternates are retained only as audited evidence.  They have the same
task and same cost, so selecting more than one would not add score.  `r001` is
preferred because it uses the direct five-way diagonal with no additional
summed label, and it is the SHA that completed full reachable support x4.

## Artifacts

- `candidate_inventory.json`: eight-file static, actual-cost, exact-rewrite, and known-x4 evidence
- `task013_support/*.json`: four complete support runs
- `task013_colour_proof.json`: colour-equivalence proof
- `task013_runtime_shapes.json`: all-node truthful shape trace
- `result.json` and `winner_manifest.json`: machine-readable disposition
