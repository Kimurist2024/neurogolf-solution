# Reduce nonnegative/binary carrier scan 143

- Scope: every `ReduceL1` and `ReduceSumSquare` node in all 400 immutable
  `8009.46` authority payloads.
- Profiles: 49.
- Rewrite profiled: replace the reduction operator with `ReduceSum` while
  preserving inputs, axes, attributes, output, and surrounding graph.
- Strictly cheaper profiles: 0.
- Safe adoptees: 0; projected gain: `+0.0`.

Because no replacement was cheaper, nonnegativity/binary-domain semantic
admission was unnecessary.  `scan.json` records every source producer and both
official-like zero-input profiles; `scan.py` reproduces the census.  The root
submission and score ledgers were not modified.
