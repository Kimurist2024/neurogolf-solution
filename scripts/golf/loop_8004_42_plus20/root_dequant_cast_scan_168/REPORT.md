# Unit DequantizeLinear scan 168

All 400 authority payloads contain nine `DequantizeLinear` nodes in total.
None has an all-one initializer scale, so no all-input exact
`DequantizeLinear(x, 1, 0) -> Cast(x)` candidate exists.

Safe adoptees/probes: **0**; projected gain: **+0.0**. Evidence: `scan.py`,
`scan.json`.
