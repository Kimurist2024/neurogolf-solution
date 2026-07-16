# Consecutive exact-chain scan 167

All 400 authority payloads were scanned for removable consecutive chains:
idempotent `Abs/Ceil/Floor/Relu/Round/Sign`, double
`Neg/Not/BitwiseNot`, same-target `Cast`, `Reshape` chains, and composable
`Transpose` chains.  No single-use consecutive site exists in the current
authority.

Safe adoptees/probes: **0**; projected gain: **+0.0**. Evidence: `scan.py`,
`scan.json`.
