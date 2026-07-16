# task008 strict regolf report

## Decision

**NO_STRICT_LOWER_TRUTHFUL_CANDIDATE.** No model was accepted or promoted, and this lane contributes `+0` to the score.

The current task008 authority is not an eligible sound baseline: it is shape-untruthful and it is not total on the generator domain. Algebraically equivalent initializer/constant rewrites preserve that semantic error, while the safe rewrites tested here do not lower the official cost.

## Authority guard

- archive: `submission_base_8009.46.zip`
- archive SHA-256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- extracted member: `authority/task008.onnx`
- authority SHA-256: `30abdd1f30f1aa88549edbf22c6e7a4af4fec3036fd8809812456ccb0df6e292`
- graph: 60 nodes, 16 initializers, 5,957 serialized bytes
- official profile: memory 331 + params 100 = cost 431
- generator: `inputs/arc-gen-repo/tasks/task_05f2a901.py`
- generator SHA-256: `1cc6165c9eba7aed2a60d2c5a0e13ba108a6426b544efa49411a93d6436302b3`

The root `submission.zip` and `submission_base_8009.46.zip` had the same SHA-256 at both the start and final guard. This lane did not edit `submission.zip`, `all_scores.csv`, `others/71407`, or its manifest.

## Semantic audit

The task rule moves the perforated red rectangle along the separated axis until it touches the cyan 2x2 object, preserving the red pattern. The generator can also flip or transpose the constructed case.

The authority passes all 266 known cases under four ONNX Runtime configurations, but independent fresh generation at seed `194008` gives the same result in every configuration:

| ORT configuration | right | wrong | errors | finite output |
|---|---:|---:|---:|---:|
| disable-all, 1 thread | 1,958 | 42 | 0 | yes |
| disable-all, 4 threads | 1,958 | 42 | 0 | yes |
| default, 1 thread | 1,958 | 42 | 0 | yes |
| default, 4 threads | 1,958 | 42 | 0 | yes |

This is 97.9%, not a generator-total proof. The first failure is zero-based case 88 / one-based case 89.

### Explicit reachable counterexample

`capture_counterexample.py` reconstructs case 89 by calling the generator's explicit `generate(**latent)` with:

```text
width=15, height=8, wide=4, tall=2,
redrow=0, redcol=11, cyanrow=5, cyancol=10,
rows=[0,0,0], cols=[0,1,3], flip=1, xpose=0
```

The explicit reconstruction exactly equals the seeded sample, so this is inside the generator support rather than an arbitrary hand-written input. Both default and disable-all ORT leave the isolated red cell at grid coordinate `(6,11)` instead of moving it to `(3,11)`. That produces four differing one-hot entries (background and red channels at the old and new cells). Exact input, gold, prediction, and per-mode differences are in `counterexample.json`.

The failure is consistent with the current min-position encoding missing the only red cell in the leftmost occupied column. Therefore an exact local rewrite of this graph cannot satisfy a full-input correctness requirement.

## Shape truthfulness audit

The model declares its output as int8 `[1,1,1,1]`, but every tested execution returns `[1,10,30,30]`. A probe that changes only the declared output to `[1,10,30,30]` fails both full ONNX checking and strict shape inference:

```text
ScatterND inferred shape and existing shape differ in dimension 1: (1) vs (10)
```

Runtime tracing of 32 fresh cases found 60 tensors and zero nonfinite values. Important declared/runtime-hidden tensors include:

| tensor | observed runtime shape(s) | max bytes |
|---|---|---:|
| `input_cloak` | `[1,10,30,30]` | 36,000 |
| `input_f16_hidden` | `[1,10,30,30]` | 18,000 |
| `input_i8_hidden` | `[1,10,30,30]` | 9,000 |
| `red_crop_i8_4` | `[1,1,3,5]` or `[1,1,5,3]` | 15 |
| `grid3_f16` | `[2,2,3,5,3]` or transposed | 360 |
| `indices2x2xhwx5_f16` | `[2,2,3,5,5]` or transposed | 600 |
| `indices_cloak_f16` | `[2,2,3,5,4]` or transposed | 480 |
| `indices_i64_hidden` | `[2,2,3,5,4]` or transposed | 1,920 |
| `updates2x2hw_i8` | `[2,2,3,5]` or transposed | 60 |
| `output` | `[1,10,30,30]` | 9,000 |

For the current AffineGrid-to-ScatterND representation, the truthful int64 index carrier alone costs at least 1,920 bytes. With the existing 100 parameters, that representation has a lower bound of at least 2,020 before counting any other activation. This is a representation-specific lower bound, not a proof against every possible ONNX construction.

## Strict-lower probes

All safe scalar reconstruction probes pass full checker, strict `data_prop=True` inference, standard-domain audit, Conv UB0 audit, known 266 cases, four ORT configurations, and nonfinite-output checks. None is strict-lower:

| probe | SHA-256 | memory | params | cost | result |
|---|---|---:|---:|---:|---|
| authority | `30abdd1f30f1aa88549edbf22c6e7a4af4fec3036fd8809812456ccb0df6e292` | 331 | 100 | 431 | unsound / shape-untruthful |
| derive `two_i8` as `1+1` | `39620c960bdeb07ee1123be909ab96bc5b4991685b4e59c3b5e94385643ab7ee` | 332 | 99 | 431 | cost-neutral |
| derive `three_i8` as `1+2` | `51eae341a4e9d85b80410d4d737dc37ec95130bbe0e5802394ee3bd8e77e9026` | 332 | 99 | 431 | cost-neutral |
| derive `five_i8` as `2+3` | `6ee829ca223e11751f0c3a986a980d8bac8a66419fd24f857654138436c4759a` | 332 | 99 | 431 | cost-neutral |

Each removes one initializer element but creates one byte of live int8 activation, so `memory + params` is unchanged.

## Requested search areas

- **Initializer aliasing / i8 scalar sharing:** there are no exact duplicate initializers. Reconstructing a scalar saves one parameter and costs one activation byte. The three direct probes confirm the neutral result. Removing the int32 zero anchor was already known to be exact but exposes 24 additional activation bytes, changing cost 431 to 454.
- **Sign vector:** reconstructing the four-element sign vector with `Neg`/`Concat` saves at most four parameters but requires larger intermediate/output activations, so it cannot beat the authority under this cost model.
- **Input shape:** replacing the four-element stored shape with `Shape(input)` saves four parameters but materializes a 32-byte int64 vector.
- **Theta / affine-size:** `theta_flat_i8` (24 B), reshaped theta (24 B), theta float16 (48 B), affine size int8 (5 B), and affine size int64 (40 B) are required by the consuming schemas in the current lineage. Dynamic construction or extra concatenation does not reduce the live-byte total.
- **Concat / affine fusion:** the consumer schemas still require the same complete tensors. Extra `Concat`, `Reshape`, or `Cast` outputs retain equal or greater live bytes; ONNX nodes cannot encode the nested expressions without materializing them.
- **Quantize scale / zero-point absorption:** omitting `qmax_zp_i8=12` changes QuantizeLinear output to uint8, incompatible with the int8 downstream graph. A correcting Cast costs at least one byte at each affected scalar. Reusing a small existing zero point in this particular exponential/log encoding requires a large intercept shift and pushes the current float16 code range outside its usable finite envelope. The positive `q5_scale_f16` and negative `qneg_scale_f16` serve non-reciprocal paths; aliasing them needs additional Neg/reciprocal outputs and loses memory.
- **Position vectors:** trimming the 30-element position initializers needs padding or dynamic reconstruction, whose activation cost exceeds the saved parameters. The prior sparse-initializer route fails strict rank/type requirements.
- **Exact generator-domain Quantize/Log code:** the code is not exact over the generator support, as the explicit case-89 counterexample proves. Fixing the omitted extremal red position is a semantic redesign, not a sound parameter shave. Existing ground-up references (`scripts/golf/scratch_codex/task008/REPORT.md`) were also above 431 and did not provide a truthful strict-lower total solution.

## Gate summary

- full checker: pass for authority and three scalar probes
- strict shape inference with `data_prop=True`: pass for authority and three scalar probes
- standard domains: pass
- Conv bias/UB0 audit: no findings
- known cases: 266/266 in default/disable-all, threads 1/4
- independent fresh: authority 1,958/2,000, hence fail generator-total correctness
- runtime errors: 0
- nonfinite output values: 0
- all-intermediate nonfinite values in trace: 0
- truthful declared/runtime shapes: fail
- strict-lower cost: none
- accepted candidate / ZIP: none

Reproduction scripts and machine-readable evidence are `build_probes.py`, `audit.py`, `capture_counterexample.py`, `build.json`, `audit.json`, and `counterexample.json` in this directory.
