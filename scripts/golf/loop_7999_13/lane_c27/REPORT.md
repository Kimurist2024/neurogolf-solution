# C27 strict optimization report

## Outcome

No candidate is eligible for adoption. Verified projected gain is **+0.0**.
The exact Wave15 members remain task184 cost 421 and task377 cost 409. No root
ZIP, CSV, score pointer, or shared handcrafted artifact was changed.

The baseline members are byte-identical to the original 7999.13 members:

| task | cost | SHA-256 |
|---:|---:|---|
| 184 | 421 | `156fe12922d290876f63c210f9cec8252e308e1af0512e309cb2fc6fad8928fc` |
| 377 | 409 | `ce349264110182cfdb5c8dbbefcd4bb36db5c1f83d143e38a4514f32d03c4318` |

## task184

The raw generator rule is to recover the dominant color of every noisy
rectangular patch in a 2x2, 2x3, 3x2, or 3x3 patch grid. The exact incumbent
finds the all-background separator rows and columns, samples one color per
patch, and pads the small color grid back to the fixed output carrier.

The two closest historical complete-known models both measure cost 422. Their
only extra parameter is `C23_i64=[22]`, used to compute the fixed crop length
23. Replacing that Add by the exactly equivalent
`Shrink(sh0, bias=-22, lambda=0)` removes the parameter and returns cost 421,
but does not beat the incumbent.

More importantly, all three forms are structurally ineligible:

- DISABLE_ALL known: 169/169, errors 0.
- Default ORT: session construction fails at CenterCropPad because a one-value
  shape is attached to two axes.
- Runtime trace: six declared/actual shape mismatches. For example `hid` and
  `input16` are declared `[1,1,1,1]` but run as `[1,10,30,30]`, while
  `rowhid` is declared `[1,1,1,1]` but runs as `[1,10,23,23]`.
- The one-case true intermediate footprint is 64,961 bytes, versus the cloaked
  reported memory 389. Correcting the declarations therefore cannot produce a
  truthful model below cost 421.

Candidate `task184_r06_shrink421.onnx` is rejected; fresh-5000 cannot rescue a
non-cheaper model with false shapes and a default-runtime session failure.

## task377

The raw rule is to read the nested rectangle colors and render the corresponding
odd-sized concentric square. Historical exhaustive work already established
that all known-correct CSE alternatives profile at cost 431 or higher and that
the two cost-408 coordinate deletions fail complete known data.

This lane found one new exact parameter identity. The existing 5x5 upper
bidiagonal `diff5` has total sum exactly one. Reshaping it to `[1,5,5]` preserves
its 25 parameters, supplies the old singleton `k` dimension, and lets its two
matrix axes be contracted as a scalar-one witness in the three earlier
Einsums. It also remains the same matrix in the final free-output Einsum.
Consequently `one_vec_f16=[1]` can be removed, giving **409 -> 408** with no
new operation and no giant Einsum.

The probe is mathematically exact and passes the full checker, strict symbolic
shape inference, and complete DISABLE_ALL known data 266/266 with errors 0.
It is nevertheless rejected:

- Default ORT: 0 correct, 0 wrong, **266 runtime errors**.
- The runtime trace itself terminates on the inherited false-shape buffer reuse
  (`{1,5,1}` versus `{1,5,10}`). The graph also declares the final output
  `[1,1,2,2]` although it runs as `[1,10,30,30]`.
- Thus the apparent +`ln(409/408)=0.0024479816` depends on the incumbent's
  shape cloak and violates the dual-ORT error-zero and truthful-shape gates.

Candidate `task377_diff5_witness408.onnx` must not be merged.

## Evidence and admission decision

All audited models pass ONNX full checking and strict data-propagating symbolic
inference, have no banned operations, nested graphs, foreign domains, lookup
red flags, giant Einsums, or unsafe Conv biases. Runtime shape tracing and both
ORT modes are decisive here. Machine-readable evidence is in
`evidence/model_audit.json`; final disposition is in `winner_manifest.json`.

Fresh-5000 was not started because no candidate passed the earlier mandatory
prerequisites of strict cost improvement, truthful static shapes, and zero
runtime errors under both ORT configurations.
