# NeuroGolf LB-verified checkpoint 8006.61

This directory freezes the current LB-verified champion without modifying the
protected root pointers.

- LB score: `8006.61`
- ZIP SHA-256: `9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`
- ZIP MD5: `c90b23f514fe47c36f1d032bd0924662`
- Source: `submission_base_8006.61.zip`
- Root parity: byte-identical to `submission.zip`
- Members: 400 ONNX files; archive integrity PASS
- Conv/ConvTranspose/QLinearConv bias-length UB: 0
- Order-sensitive final 33 members: unchanged

Relative to LB 8005.17, the champion changes seven unique tasks:
`013`, `070`, `158`, `254`, `267`, `323`, and `379`.  The first LB wave
adopted `158/254/267/323` and reached 8006.47.  The second wave adopted
`013/070/158/379` and reached 8006.61.  The `candidates/` directory contains
the exact seven payloads from the final champion.

The previous files directly under `others/71403/` are retained as historical
pending-LB Wave15 evidence.  Use this `lb_verified_8006.61/` subdirectory as
the immutable current checkpoint.
