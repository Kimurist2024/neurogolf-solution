# A36 task158 exact permutation-bank reuse

Against `submission_base_8000.46.zip` SHA-256
`74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534`,
task158 has one strict winner: cost **7627 -> 7615**, projected score gain
**+0.001574596835**.

## Generator bounds and shape floor

`task_6aa20dc0.py` draws width in 15..25 and height in
`{width-1,width,width+1}`.  These are exactly 33 reachable `(height,width)`
shapes, ranging from 14x15 through 26x25.  A magnification-one sprite can be
placed at row 23 in height 26, making its bottom endpoint-block top row 25;
similarly width 25 reaches endpoint-block column 24.  Both have stride-two tile
index 12.  The existing anchor score therefore needs all **13x13** tiles; no
anchor row or column is unreachable.  The full output can likewise paint row
25 and column 24, so the **26x25** paint seed cannot be cropped.

The remaining list shapes are also generator-tight: four sprites produce up
to eight endpoint anchors; after removing the visible key, at most three
objects remain; the pairing problem is therefore exactly 3x3 with six
permutations; and the key has up to six non-endpoint cells to paint.  The
coordinate moment operands must stay length 30 because they contract directly
with the competition's `[1,10,30,30]` input.  Cropping that input would
materialize a much larger counted tensor.

## Exact rewrite

The baseline stores every one of the six 3-element permutations twice:

- `more_perm_mask[6,3,3]`, used to score all matchings; and
- `more_perm_indices[6,3]`, used to reorder the winning q rows and columns.

Every row and column of each mask contains exactly one `1`.  The candidate
keeps the mask bank, gathers the winning 3x3 mask, and applies its rows directly
to `q_row` and `q_col` with two small two-input Einsums
(`bij,bj->bi`).  This is exact one-hot selection, not a learned lookup or a
giant contraction.  It removes all 18 index parameters.  The selected mask is
18 fp16 bytes versus the former 12-byte int32 selected-index tensor, so memory
increases by 6 while parameters fall by 18: net cost reduction 12.

No other node shape, Conv/QLinearConv topology, anchor crop, paint seed, or
generator rule changes.  The dynamic anchor Conv weight is inferred as
`[1,10,3,3]` and its bias is `[1]`; the QLinearConv weight and bias also both
have one output channel.  Thus both bias lengths are exact and there is no
Conv-family UB.

## Gates

- known corpus: 266/266, errors 0 under both default ORT and
  `ORT_DISABLE_ALL`;
- known raw outputs: 266/266 bitwise equal to the baseline in both modes;
- fresh generator: 5000/5000, errors 0 in both modes, with raw outputs bitwise
  equal to the baseline on all 10,000 mode/case comparisons;
- all 33 generator-reachable `(height,width)` shapes occurred in that fresh
  run;
- margin: no value in `(0,0.25)`, minimum positive value 1.0;
- full ONNX checker and strict shape inference/data propagation: pass;
- truthful runtime shapes: 179/179 declared intermediates match runtime;
- standard opset only, no functions, nested graphs, sequences, sparse
  initializers, banned ops, giant tensors, lookup table keyed by examples, or
  shape cloaking;
- external validator: valid/preflight pass, known 266/266, cost 7615,
  156/156 executable arbitrary differential cases raw-equal, verdict
  `ACCEPT_STRICT`.

The isolated ZIP has 400 unique models, valid CRC, unchanged member order,
archive comment and metadata, and changes only `task158.onnx`.  Its SHA-256 is
`228db6f07d5e7416525afacb12c39aa111f94da0967989ff0b2bf6532626e865`.
Because an old task158 processing-error incident has no recorded model hash,
this ZIP remains separately identifiable even though the candidate contains
none of the known processing-error structures.

