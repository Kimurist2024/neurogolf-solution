# 8018.91 urgent cost-400..500 lane

## Strict winner

- `task275`: cost `428 -> 419`
- score gain: `+0.021252275660`
- candidate: `candidates/task275_POLICY90_cost419_c7ddaab77f6d.onnx`
- SHA-256: `c7ddaab77f6da011a99d233775ab02964f1a5e714f4dbb02045d1ecdda57c8e2`
- local gold: exact pass
- independent official-module gold: exact pass
- ONNX full checker, strict inference, static shapes: pass
- minimum positive raw margin: `38415.54296875`
- fresh seed `408120275`: `2000/2000`, errors/nonfinite/shape/small-positive all zero
- fresh seed `408220275`: `2000/2000`, errors/nonfinite/shape/small-positive all zero

One-member probe ZIP:
`submission_PROBE_task275_cost419.zip` (only `task275.onnx` differs from
`submission_base_8018.91.zip`).

## Coverage

All 18 authority tasks in cost 400..500 were scanned across loose-model
history, ZIP-member history, exact simplifiers, and low-cost transfers:

`008, 025, 062, 102, 112, 134, 156, 184, 250, 268, 270, 275, 308, 324,
333, 354, 374, 377`.

The other 17 tasks produced no candidate satisfying the absolute gate.
In particular, archived `task134@320/322` passed known gold but failed both
independent fresh streams (`1999/2000` each), so neither is admitted.

## Authority protection

- authority: `submission_base_8018.91.zip`
- authority SHA-256:
  `e43865760ec8807fbb217fba718226ca6b86d9128b911479214e3252b9f9e091`
- this lane did not modify root `submission.zip`, `all_scores.csv`, or
  `best_score.json`; probe guards are recorded in `probe_manifest.json`.
