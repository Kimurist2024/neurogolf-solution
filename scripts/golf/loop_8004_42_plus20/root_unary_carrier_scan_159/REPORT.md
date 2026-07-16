# Unary carrier scan 159

All400 immutable payloads were scanned for scalar-initializer binary operators
with unary equivalents: `Div(1,x)->Reciprocal`, `Sub(0,x)` or
`Mul(-1,x)->Neg`, and `Max(0,x)->Relu`.

Only36 `Sub(0,x)->Neg` sites existed. Every candidate fails full/strict ONNX
validation: almost all inputs use unsigned integer types unsupported by Neg;
the remaining malformed authority lineage fails strict downstream inference.
No cost/runtime candidate exists.

Safe adoptees/probes: **0**; projected gain: **+0.0**. Evidence: `scan.py`,
`scan.json`.

