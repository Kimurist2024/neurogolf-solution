# Greater/Where activation fusion scan 263

All 400 authority graphs contain 1,039 `Where` and 131 `Greater` nodes.  The
scan looked for exact, same-shape forms of:

- `Where(x > alpha, x, 0) -> ThresholdedRelu(alpha)`;
- the `alpha=0` specialization to `Relu`; and
- `Where(x > 0, x, slope*x) -> LeakyRelu(slope)`.

No graph contains one of these producer/consumer patterns with static matching
shapes and initializer-backed constants.  Safe adoptees: **0**; projected
gain: **+0.0**.  `scan.py` and `scan.json` contain the reproducible census.
