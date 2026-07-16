# Root integer carrier scan 242

## Result

REJECT: the all-400 authority scan produced no candidate that is both strictly
lower-cost and authority-equivalent under the required ORT configurations.
Nothing is eligible for staging or submission.

Authority: `submission_base_8009.46.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.

## Full census

- 400 authority members inspected; 393 passed strict ONNX shape inference with
  data propagation. The seven authority-side strict failures were tasks 018,
  112, 117, 170, 243, 245, and 397, and were excluded fail-closed.
- 1,176 Cast nodes inspected. There were zero same-dtype Casts.
- One Cast had only Cast/CastLike consumers: task173
  `UINT8 x_pair -> BOOL x_pat -> UINT8 fam_x`. It is a required nonzero-to-1
  normalization, not a removable conversion.
- 568 closed INT64 carrier components and 257 closed INT32 carrier components
  had exact integer interval proofs.
- The resulting search covered 2,219 target transformations: 135 failed the
  proved target range, 1,839 failed strict schema/shape inference, 177 failed
  ORT session construction, and 68 passed checker + strict inference + ORT.
- Of the 68 valid transformations, 67 were INT64-initializer-to-INT32 changes
  with unchanged official cost. No INT8 or INT16 target survived the strict
  schema gates.
- Float64 occurred only in task025 (eight DOUBLE initializers). Two are not
  exactly representable as float32 (`vsgn` and `outscale`), and the graph output
  itself is DOUBLE. It therefore has no strict-exact float32 initializer or
  internal-carrier reduction that preserves the output signature.

No input or output dtype was changed. No shape cloak was used.

## Only strict-lower lead: task233

Static proof found one exact-looking Cast removal:

- `ArgMax(slotmatchu, axis=1)` emits INT64 indices in the statically proved
  interval `[0, 6]`.
- The authority Casts those indices to INT32 as `ci_sel5`, used only by three
  Gather index inputs.
- The candidate deletes that Cast and connects the INT64 ArgMax result directly
  to the three Gathers. The model I/O signature remains exactly
  `FLOAT[1,10,30,30] -> BOOL[1,10,30,30]`.
- Cost falls from memory/params/cost `6991/317/7308` to
  `6971/317/7288`, projected `ln(7308/7288) = 0.002740478558103065`.

The candidate passes full ONNX checker and strict shape inference, but fails
the required known-case four-configuration authority audit:

| ORT config | authority-equal/error-equivalent | errors (authority/candidate) |
|---|---:|---:|
| DISABLE_ALL, 1 thread | 266/266 | 0/0 |
| DISABLE_ALL, 4 threads | 266/266 | 0/0 |
| ENABLE_ALL, 1 thread | 264/266 | 2/0 |
| ENABLE_ALL, 4 threads | 264/266 | 2/0 |

The divergent cases are `arc-gen[139]` and `arc-gen[245]`: optimized authority
execution raises a Reshape runtime error while the candidate succeeds. This is
an error-behavior change, so the lead is rejected even though all successful
pairs have equal raw outputs and no nonfinite or `(0, 0.25)` values. Since it
failed the known gate, it is not a survivor and the 2 seeds x 5000 fresh audit
is intentionally skipped fail-closed.

## Optimizer-barrier follow-up

Twelve exact INT64 barriers were tested after the removed Cast: Identity, Abs,
Add-zero, Mul-one, Div-one, Max-zero, BitwiseOr-zero, BitwiseXor-zero,
same-shape Reshape, same-shape Expand, rank-1 Transpose, and Clip-min-zero.
Every barrier passes structural gates but costs `7011/317/7328`, exceeding the
authority cost. Thus none is eligible for runtime promotion.

## Evidence

- `scan.py` / `scan.json`: reproducible all-400 census and candidate scan.
- `audit.py` / `audit.json`: task233 structural and known four-config audit.
- `barrier_scan.py` / `barrier_scan.json`: exact barrier cost/known screen.
- `candidates/task233_integer_carrier.onnx`: rejected probe only, SHA-256
  `b4d80db94e46884ca570fff0a35a3030a766b883966a28100b619333a9f44cf0`.

