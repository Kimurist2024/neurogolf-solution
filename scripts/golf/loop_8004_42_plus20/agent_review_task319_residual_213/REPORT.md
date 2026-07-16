# task319 residual independent fail-closed review 213

## Decision

**PASS — exact inherited-authority pass-through; accept the cost-975 candidate.**

- authority: `others/71407/task319.onnx`
  - SHA-256 `ade6b708b4ee6a0ba65d19e4182748750514435b3b8a005289582154b7208fd4`
- candidate: `agent_task319_residual_210/candidates/task319_combined_best_local.onnx`
  - SHA-256 `a4e0531b0a3dc08355d429ba9a049f8dbd076b203a8ddb8f88c635bedf9f31cd`

The candidate intentionally retains the authority's metadata cloak and is not
shape-truthful.  Independent tracing found the exact same 26 declared/runtime
shape-mismatch signatures in both models under both disabled and default ORT
optimization, with **zero new mismatch**.  Acceptance is therefore restricted
to the inherited-cloak policy and is supported by complete-domain algebraic
proof of all three edits below.

## Official profile

The faithful repository scorer was rerun independently.

| model | memory | params | cost | task score |
|---|---:|---:|---:|---:|
| authority | 840 | 138 | **978** | 18.114490329965182 |
| candidate | 848 | 127 | **975** | 18.117562529002154 |

Exact gain: `ln(978/975) = 0.003072199036970059`.

Both models remain 267/267 on the known train, test, and arc-gen rows.

## Independent raw-output audit

Fresh seeds were `319213041` and `319213079`, with 1,500 successful generator
instances per seed and no generation error.

| ORT configuration | known raw equal | fresh raw equal | runtime errors | nonfinite |
|---|---:|---:|---:|---:|
| disable-all, threads=1 | 267/267 | 3000/3000 | 0 | 0 |
| disable-all, threads=4 | 267/267 | 3000/3000 | 0 | 0 |
| default, threads=1 | 267/267 | 3000/3000 | 0 | 0 |
| default, threads=4 | 267/267 | 3000/3000 | 0 | 0 |

Raw arrays were compared before thresholding.  Candidate minimum positive
output was 1 and maximum was 127.  Both models were 2924/3000 against the fresh
truth; all 76 misses are inherited authority behavior, while authority and
candidate remained raw-identical on every case.

A separate 256-case intermediate trace in disabled and default modes checked
the ArgMax index, removed uint8 Cast, background equality mask, reduced scalar
condition, transposed background mask, base weights, final weights, and final
output.  All eight relations had zero failures and no runtime error in either
mode.

## Complete-support proof of the three rewrites

### 1. Remove the ArgMax uint8 Cast

The fixed input has ten channels.  `CenterCropPad` only changes axes 2 and 3,
`CastLike` preserves shape, and `ReduceL1` reduces only axes 2 and 3.  Thus the
following `ArgMax(axis=1)` always returns an int64 index in `[0,9]`, regardless
of the authority's intentionally false declared metadata.

Casting `[0,9]` to uint8 is injective.  Comparing the original uint8 ramp to
the cast index is therefore identical to comparing an int64 ramp directly to
the int64 index.  All ten possible indices were exhaustively checked with zero
failure.

### 2. Reduce fixed `[1,1,2]` equality directly to scalar

The old path reduces axis 2 with `keepdims=1`, then squeezes every singleton
axis.  The candidate reduces all axes with `keepdims=0`.  Axes 0 and 1 are
fixed singletons, so both operations compute the same conjunction of the two
bits and return the same scalar.  All four boolean assignments were checked
with zero failure.

### 3. Build terminal background weights from the existing mask

`safe_name_29 = Equal([0..9], ArgMax)` contains exactly one true channel.
Transposing `[1,10,1,1]` to `[10,1,1,1]` produces the Conv-weight background
mask.  `Where(mask,0,1)` therefore exactly equals scattering zero at ArgMax
into ten ones.

The second target-color Scatter is protobuf-identical and runs after the
background update in both graphs, so its value 2 has the same priority even
when target and background indices coincide.  All `10 × 10 = 100`
background/target index pairs were checked with zero failure.

These are local identities over every runtime input satisfying the model's
fixed input shape, not sample-dependent observations.  They prove that the
candidate preserves the authority's raw output on complete generator support,
including the authority's non-injective/private-zero cases.

## Protobuf scope and structural gates

The complete protobuf diff is limited to the intended edits:

- nodes removed: `safe_name_28`, `cond1`; added: `bg_mask_w`;
- common nodes changed: `safe_name_29`, `cond1s`, `w_base2`;
- initializers removed: ten-one base and one-zero update; changed: color ramp
  from uint8 to int64;
- value-info removed/added only for the corresponding removed/added tensors;
- graph headers, I/O, every other node, initializer, and value-info are byte
  identical.

Both models pass full ONNX checker, strict inference with `data_prop=True`,
canonical I/O, positive static declared metadata, standard-domain, banned-op,
Sequence, nested-graph, function, sparse/external-data, finite-initializer,
size-limit, and Conv/QLinearConv bias-UB0 gates.

On 64 independent fresh cases in both optimization modes, authority and
candidate each exposed the exact same 26 inherited mismatch signatures.
Candidate-minus-authority and authority-minus-candidate sets were both empty;
runtime errors and nonfinite intermediate values were zero.

Reproducible evidence is in `audit.py` and `audit.json`.  This review did not
edit the root submission, score ledger, staged model, or lane-210 artifacts.
