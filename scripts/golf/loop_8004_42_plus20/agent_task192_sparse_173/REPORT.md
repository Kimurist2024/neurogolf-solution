# task192 SparseTensorProto lane (173)

## Decision

**REJECT ALL / NO APPLY.**  No sparse-initializer variant passes the official
full checker and strict shape inference, so none is eligible for the staged
`others/71407/task192.onnx` replacement.  Root submission files were not
changed.

## Baseline and theoretical saving

- source: `others/71407/task192.onnx`
- source SHA-256: `51a7d65491f300b91243ef8b522c00d2b699eb07615a5d64c5694de33e4d0490`
- official-compatible profile: memory 248 + params 947 = **1195**
- dense `adj`: float32 `[30,30]`, 900 values, 88 nonzero values
- if direct sparse consumption were scorable: params 135, projected cost 383,
  projected score gain `ln(1195/383) = +1.137866`

## Exhaustive metadata/indices matrix

Both legal COO encodings were generated and reconstructed bit-identically:

- flattened indices: int64 `[88]`
- coordinate indices: int64 `[88,2]`

Each encoding was tested with five metadata modes: none, dense
`graph.value_info`, sparse `graph.value_info`, dense `graph.input`, and sparse
`graph.input` (10 variants total).  Sparse initializer names were deliberately
made fixed points of `scoring.sanitize_model`; the rejection is therefore not
a sanitizer binding bug.

Results:

| Metadata (both COO forms) | Basic checker | Full checker / strict data-prop | ORT raw + sanitized, default + disabled | Competition profile |
|---|---:|---|---|---|
| none | pass | fail: Einsum input 4 rank inferred as 0 | pass, bit-identical | reject |
| sparse value_info | pass | fail: Einsum input 4 rank inferred as 0 | pass, bit-identical | reject |
| dense value_info | pass | fail: tensor/sparse_tensor type mismatch | pass, bit-identical | reject |
| dense graph input | pass | fail: tensor/sparse_tensor type mismatch | pass, bit-identical | reject |
| sparse graph input | pass | fail: Einsum input 4 rank inferred as 0 | fail: initializer/use type mismatch | reject |

Thus the old `root_sparse_nondot_scan_149` failure was not caused by malformed
flattened indices.  Two-dimensional coordinate indices fail identically.
ONNX's Einsum schema accepts only `tensor(...)` types and contains no
`sparse_tensor(...)` type constraint.  ORT happens to materialize a sparse
initializer as a dense tensor for ordinary execution, but the scorer first
runs ONNX full checking/shape inference, where this operand has no usable
tensor rank.  Adding dense metadata creates an explicit type conflict; adding
sparse metadata preserves the unusable sparse rank.

## Sparse Constant control

`Constant(sparse_value=...)` was tested with both COO forms.  Both controls:

- pass full checker and strict data-prop;
- pass raw/sanitized ORT at default and disabled optimization;
- are raw-output bit-identical to the source on the runtime probe;
- pass the complete official-compatible correctness/profile pipeline with
  `correct=true`.

However each profiles at memory 3848 + params 135 = **3983**, a regression of
2788 versus 1195.  Materializing the dense `[30,30]` Constant output adds 3600
memory elements, so this supported path is not competitive.

## Deep-audit policy

Known4 and fresh10000 were intentionally not run: the required prerequisite
was an official-profiled cost below 1195, and no candidate met it.  Running a
large behavioral audit cannot cure a deterministic checker rejection.  The
exact source tensor, all generated candidates, individual gate errors, ORT
hashes, and profiler results are preserved in `probe_results.json`.

## Integrity

- `submission.zip` SHA-256 remained
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
- staged source SHA-256 remained
  `51a7d65491f300b91243ef8b522c00d2b699eb07615a5d64c5694de33e4d0490`.
- No candidate was copied into `others/`, the root task set, or a submission.

