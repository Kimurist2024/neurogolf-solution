# Schema-default attribute scan 164

All 400 authority payloads were scanned against their active ONNX operator
schemas.  Every explicitly stored attribute whose value exactly equals the
schema default was removed, both as a per-task combined variant and one at a
time.  The scan profiled 1,171 variants.  Sixty-five inherit structural or
strict-inference failures; none of the 1,106 valid variants is cheaper under
the competition profiler because these attributes do not contribute to the
scored memory/parameter total.

Safe adoptees/probes: **0**; projected gain: **+0.0**. Evidence: `scan.py`,
`scan.json`.
