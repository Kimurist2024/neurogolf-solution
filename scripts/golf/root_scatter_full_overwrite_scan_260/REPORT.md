# Scatter full-overwrite exact scan (lane 260)

## Result

No safe strict-lower survivor was found.  The pinned `submission.zip`
(`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`)
was not modified.

## Exhaustive scan

- Scope: all 400 authority models.
- Scatter-bearing tasks: 74.
- ScatterND nodes: 22.
- ScatterElements nodes: 117.
- Total Scatter nodes: 139.
- Statically proved full-overwrite nodes: 1.
- All-use removable constant-index groups: 0.
- Strict-lower constructed candidates: 1 (`task300`), rejected.

Every constant index is normalized against its target dimension before the
proof.  ScatterND expands every update prefix into its complete target block
and rejects duplicates, missing blocks, and maps outside Identity, Reshape, or
Transpose.  ScatterElements checks every axis-line, requires the same complete
permutation on all lines, and validates the inverse Identity/Slice/Gather map.
Shared initializers are deleted only when every use is a selected, proved index
use; otherwise they remain and receive no parameter-saving credit.

The 138 non-proved nodes broke down as follows:

- 91: indices were runtime-computed rather than initializers.
- 39: `reduction` was not `none`.
- 6: ScatterElements did not overwrite the full data shape.
- 2: ScatterND had too few prefix blocks for a complete overwrite.

Eight synthetic finite-map checks cover ScatterND Identity/Reshape/Transpose,
ScatterElements Identity/reverse-Slice/non-arithmetic inverse-Gather, and
duplicate rejection for both operator families.  All eight pass.

## Sole static hit: task300

The declared graph makes ScatterND node 25 look like a complete overwrite:

- declared data/output: `[1,1,1,1]`;
- indices `bg_index`: `[0]`;
- updates `bg_update`: shape `[1,1,1]`, value `-1`;
- proposed replacement: `Reshape(bg_update, [1,1,1,1])`.

`bg_index` is also the shape input of node 6, so it cannot be removed.  The
candidate instead adds four shape parameters and prunes one now-dead node.  Its
official measured cost is 175 -> 169 (memory 120 -> 110, params 55 -> 59).

This is a shape-cloak false positive, not a semantic full overwrite.  Runtime
profiling of the authority gives `sel_w_i8` and `dyn_w` shape `[10,1,1,1]`, so
index `[0]` replaces only the first of ten prefix blocks.  The replacement
collapses all ten blocks to one.  Consequently:

- full ONNX checker: pass;
- strict shape inference with data propagation: pass;
- structural audit and Conv-bias UB0: pass;
- official known correctness: fail;
- runtime-shape truthfulness: fail (four mismatches remain in the candidate;
  the authority has seven, including `sel_w_i8` and `dyn_w`);
- known corpus, disable-all/default x threads 1/4: 0/267 raw-equal,
  0/267 threshold-equal, 0 runtime errors in each configuration.

Fresh 2 x 1000 testing was correctly skipped because the only strict-lower
candidate failed official, runtime-shape, and known4 gates.

Machine-readable evidence is in `scan_result.json`; the rejected model is kept
under `candidates/` for reproducibility.

