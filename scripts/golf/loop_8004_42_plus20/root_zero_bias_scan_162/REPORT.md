# Zero optional-bias scan 162

All 400 authority payloads were scanned for all-zero optional bias inputs on
`Conv`, `ConvTranspose`, `QLinearConv`, and `Gemm`.  Only two eligible sites
exist, both shared scalar biases on task233 `QLinearConv` nodes.  Omitting
either input passes the structural analysis, but the initializer remains live
through the sibling node and the competition cost is unchanged (7308 -> 7308).

Safe adoptees/probes: **0**; projected gain: **+0.0**. Evidence: `scan.py`,
`scan.json`.
