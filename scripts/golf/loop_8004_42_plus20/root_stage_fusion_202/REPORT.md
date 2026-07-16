# Active-stage exact fusion rescan

## Outcome

No additional strict-lower model was found. The active 71407 set remains at
14 models and projected gain `+0.388111033097`.

## Coverage

All 14 active descendants were independently profiled under 21 exact ONNX
optimizer pass sets (294 task/pass profiles). The passes cover MatMul/Gemm,
Conv bias/BN/Pad, QKV, common-subexpression and idempotent elimination,
shape/slice simplification, Reduce/Unsqueeze, Einsum/MatMul, Where/Add, and a
combined fixed-point pass.

- changed profiles: `148`
- strict-lower profiles: `0`
- emitted candidates: `0`

This is distinct from the earlier 400-member authority scan: it checks the
later task158/task192/task226/task245/task328 and other staged descendants
themselves for newly exposed follow-on fusions.

No root ZIP, score ledger, or staged ONNX was modified. Machine-readable
results are in `scan.json`.
