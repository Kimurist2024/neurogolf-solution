# NeuroGolf improvement checkpoint

> Current LB-verified checkpoint: `lb_verified_8006.61/` (LB 8006.61,
> SHA-256 `9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`).
> The files described below are retained as historical pending-LB Wave15
> evidence and are not the current champion.

This directory freezes the currently admitted pending-LB improvements on top
of the immutable LB-verified `submission_base_8005.17.zip`.

- Baseline score: `8005.17`
- Projected score: `8006.483532199815`
- Projected gain from baseline: `+1.3135321998151994`
- Projected gain from the requested `8004.42` origin: `+2.0635321998151994`
- Pending-LB replacements: `task013`, `task158`, `task254`, `task267`, `task323`, `task333`
- Already fixed in the baseline: `task226` cost `372`
- Explicitly excluded: `task009` (recorded LB-black), `task036`
  (cost `1428` would regress from the actual baseline cost `325`)

`submission.zip` contains exactly 400 ONNX members and changes only the six
pending-LB tasks. Archive order, comment, member metadata, integrity, and the
full Conv-family bias audit pass. See `MANIFEST.json` and
`FULL_ZIP_AUDIT.json` for hashes and machine-readable evidence.

The `candidates/` directory contains exact copies of the six replacement ONNX
payloads. Candidates still under investigation at checkpoint time are not
included.
