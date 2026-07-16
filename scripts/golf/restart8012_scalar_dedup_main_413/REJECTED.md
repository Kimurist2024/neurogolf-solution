# task349 scalar-dedup rejection

Attempted rewrite: replace the rank-4 initializer `max29_rank4_i8` with the
existing scalar `max29_i8` at the `Min(side_i8, 29)` node.

Rejected before runtime. `onnx.checker.check_model(..., full_check=True)`
reported that downstream `Concat` inputs no longer had the same rank: the
scalar input made shape inference classify `beam_end_scalar_i8` as rank 0
instead of rank 4. The rank-4 constant is therefore load-bearing for the
authority graph's static shape proof. No candidate was emitted or admitted.
