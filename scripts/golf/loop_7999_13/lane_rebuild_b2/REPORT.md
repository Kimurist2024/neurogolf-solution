# Lane B2 final report

Baseline: **7999.13**. The exact submission archive (SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`) is the
cost authority; no root artifact or submission archive was modified.

## Accepted winner

`task204` has one strict-cheaper, sound candidate:

- Candidate: `winner_task204.onnx`
- SHA-256: `6e21b4c6a53d3f8a56d165e4a2e29dcc12aba66c77759e5e56ab032dff12bb3f`
- Cost: **2560 -> 2544** (-16)
- Per-task score: **17.152237462526394 -> 17.158507075539987**
- Projected gain: **+0.0062696130135933**
- Projected aggregate: **7999.136269613014**
- Visible: **268/268**
- Direct generator fresh: **3000/3000**, zero wrong/errors
- Independent random differential: **3000/3000 raw- and threshold-equal**,
  zero mismatches, zero one-sided errors, maximum absolute difference 0

The only edit is the inferred `masks5_rows30` value-info dimension 0 (`5 -> 1`).
Nodes, initializers, graph inputs/output, and opsets are unchanged. Clearing
`graph.value_info` makes the baseline and candidate serialize identically, so
the executable graph is preserved. Checker, strict shape inference with data
propagation, static-positive-shape, banned-op, nested-graph, function, sequence,
sparse-initializer, convolution-bias, and margin gates all pass. The minimum
observed margin is 2.0.

When merging later, preserve `task204` at its original archive member position
(zero-based index 370). The executable graph is unchanged, so this shave does
not remove the task's historical CenterCropPad allocator/order caveat.

## Other assigned tasks

- `task080` (3051) and `task138` (2731) are generator-sound at 3000/3000, but
  the metadata shave search and earlier sound-candidate review found no strict
  cheaper candidate.
- `task023` has exact cost 1622 (the assigned 1637 was stale) and scored only
  2503/3000 fresh; its private-zero ambiguity prevents safe reuse.
- `task187` and `task379` each scored 2996/3000 fresh. Their compact baselines
  are unsafe and known sound rebuilds are more expensive.
- `task216` passed the first 10 fresh cases, then raised a Gather out-of-bounds
  error on case 11. Its known sound rebuild is more expensive.

Only `task204` is accepted. The rejected compact baselines were not promoted,
and no historical black SHA or lookup memorization was reused.
