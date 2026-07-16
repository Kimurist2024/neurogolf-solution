# task205 / task338 sound memshave lane — 8009.46

## Outcome

One safe exact memshave was found for task205. No safe strict-lower task338
candidate exists in the inspected frontier.

The authority is the current `submission.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`,
which is byte-identical to `submission_base_8009.46.zip`.

| Task | Authority | Best candidate | Reduction | Projected score gain | Decision |
|---:|---:|---:|---:|---:|---|
| 205 | 1042 (memory 1031, params 11) | 1041 (memory 1031, params 10) | 1 | +0.0009601537 | exact winner |
| 338 | 406 (memory 404, params 2) | no truthful strict-lower | — | 0 | retain authority |

No root ZIP, score ledger, `others/`, or `artifacts/` path was modified.

## task205 exact winner

- Candidate: `candidates/task205_rowpow_selu.onnx`
- SHA-256:
  `509c1947929ab888cff4443ac5b6d808b213fa5057e1c03a2758c1717b3f9eed`
- Authority member SHA-256:
  `8a6acdc20a366ccbd32cf761285cbb2f1cbcf7d3d2ef8ea71d0fb5a3ed6f1468`
- Official profile: memory unchanged at 1031, params `11 -> 10`, cost
  `1042 -> 1041`.

The only changes are two uses of the same scalar initializer:

```text
Mul(tall_f, rowpow_thr=1.902)    -> Selu(tall_f, alpha=1, gamma=1.902)
Mul(roww_max, rowpow_thr=1.902) -> Selu(roww_max, alpha=1, gamma=1.902)
```

`rowpow_thr` is then removed. Every other node and every common initializer is
byte-identical to the immutable member.

This is an all-valid-input algebraic rewrite, not an example-fit rule change:

- `tall_f` is a sum of a cast boolean mask, so it is always nonnegative.
- `roww_max` is the maximum of an Einsum over finite nonnegative one-hot input,
  a 0/1 Hardmax result, and nonnegative count tensors, so it is nonnegative.
- For `x > 0`, ONNX Selu with `gamma=g` is `g*x`; at `x=0` it is
  `g*(exp(0)-1)=0`.

Therefore both substitutions equal the original multiplication over the full
NeuroGolf one-hot input domain. The existing task205 graph's behavior, including
its historical limitations, is unchanged.

### Verification

- Full ONNX checker: pass.
- Strict shape inference with data propagation: pass.
- Direct runtime trace: 37 tensors, zero declared/runtime mismatches, no
  non-finite values.
- Conv-family bias UB: zero findings.
- Official profiler: `1031 + 10 = 1041`.
- Known train/test/arc-gen: 266/266 under each of:
  `ORT_DISABLE_ALL` threads 1/4 and default ORT threads 1/4.
- Candidate vs authority on known: raw 266/266 and threshold 266/266 in every
  configuration, zero runtime errors.
- Fresh generator seed 12320501: raw 5000/5000 in both ORT modes; candidate and
  authority each score 4925/5000.
- Fresh generator seed 12320502: raw 5000/5000 in both ORT modes; candidate and
  authority each score 4939/5000.
- Arbitrary finite one-hot canvases, random height/width 1..30: raw 2000/2000
  in both ORT modes, zero runtime errors and non-finite values.
- Minimum positive output: 1; values in `(0,0.25)`: zero.

The fresh gold rate is deliberately disclosed: this is not claimed as a new
true-rule rebuild. Admission is based on a checked algebraic equivalence to the
already LB-white immutable authority, so the rewrite cannot add a private-zero
failure or change any hidden output.

## task205 rejected reductions

| Probe | Cost | Known result | Reason |
|---|---:|---:|---|
| box QuantizeLinear -> Cast | 1041 | 0/266 | changes quantized color weights |
| exact Selu + box Cast | 1040 | 0/266 | same semantic failure |
| exact Selu + `cm0 * -3 -> Neg(cm0)` | 1040 | 0/266 | weakens the signed selector |
| exact Selu + gain reuse from `colq_scale` | 1040 | 240/266 | tall-dependent gain is not equivalent |

Historical task205 candidates at cost 1038/1041 were not reused: their retained
fresh minima are only 97.4%/97.6%, and cheaper 937/997/1010/1015 families have
documented private-zero or fresh failures.

## task338 terminal result

The generator rule is sound and bounded: each red rectangular frame is replaced
by green in its strict interior, while width-2 or height-2 frames have no
interior. A correct implementation needs spatial neighbor/line-of-sight state.

The current cost-406 member obtains its nominal memory 404 through false
singleton declarations. Direct all-intermediate tracing fails with shape buffer
reuse errors; the declared output is `[1,10,1,1]` while normal execution produces
`[1,10,30,30]`. It is not a template for a new truthful candidate.

Removing only its fp16 `CastLike` type witness raises the official profile to
18403 and still leaves other shape contradictions. Any honest pre-output
single-channel spatial mask alone costs at least 900 bytes as bool or 1800 bytes
as fp16, already above 406. The earlier 128-unique-model task338 archive scan
also found no known-correct model below the then-authority cost 452; its closest
known-correct result was 462, with cheaper lineages cataloged as private-zero or
shape-cloaked. No task338 candidate is admitted.

## Artifacts

- `build_candidates.py`, `build_manifest.json`: immutable extraction and probe
  construction.
- `screen_candidates.py`, `screen.json`: official profiles, static gates,
  runtime-shape traces, and four-configuration known audit.
- `audit_winner.py`, `winner_audit.json`: algebraic proof, dual-ORT fresh and
  arbitrary-one-hot differential audit.
- `manifest.json`: machine-readable winner decision.

