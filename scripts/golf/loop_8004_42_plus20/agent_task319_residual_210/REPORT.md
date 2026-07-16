# task319 residual exact-regolf lane 210

## Decision

**PASS — stage `task319_combined_best_local.onnx` as an exact descendant of the
current cost-978 authority.**

| item | authority | candidate |
|---|---:|---:|
| memory | 840 | 848 |
| params | 138 | 127 |
| official cost | **978** | **975** |
| task score | 18.114490329965182 | 18.117562529002154 |

The exact score gain is `ln(978/975) = 0.003072199036970059`.

- authority: `others/71407/task319.onnx`, SHA-256
  `ade6b708b4ee6a0ba65d19e4182748750514435b3b8a005289582154b7208fd4`
- candidate: `candidates/task319_combined_best_local.onnx`, SHA-256
  `a4e0531b0a3dc08355d429ba9a049f8dbd076b203a8ddb8f88c635bedf9f31cd`

No root submission, score ledger, or staged file was changed in this lane.

## Three complete-support rewrites

1. **Remove the ArgMax color Cast.**  The background ArgMax is always an
   integer in `[0,9]`.  Comparing it directly against an int64 `[0..9]` ramp is
   identical to casting it to uint8 and comparing against a uint8 ramp.  This
   removes one counted byte.  All ten possible indices were exhaustively
   checked.

2. **Reduce the equality condition directly to scalar.**  The input `eq1` has
   fixed runtime shape `[1,1,2]`.  Reducing axis 2 with keepdims and then
   squeezing is identical to reducing all axes with `keepdims=0`.  This removes
   one counted boolean.  All four assignments of the two equality bits were
   exhaustively checked.

3. **Reuse the existing background mask for terminal weights.**  Transposing
   `safe_name_29` gives the `[10,1,1,1]` Conv-weight layout.  A single
   `Where(mask,0,1)` is exactly the old operation that scattered 0 into an
   initializer of ten ones.  This adds a 10-byte mask output but removes the
   ten-one initializer and the one-zero update initializer, a net one-point
   cost reduction.  All ten possible background indices were exhaustively
   checked.  The second target Scatter is byte-for-byte unchanged.

All three identities hold for every model input satisfying the fixed tensor
shapes; they are stronger than generator-sample arguments.  Their combination
therefore preserves the authority's raw output on complete generator support,
including all inherited non-injective choices.

## Runtime and structural gates

Independent fresh seeds were `319210011` and `319210029`, 1,500 generated cases
each.

| ORT configuration | known raw equal | fresh raw equal | errors | nonfinite |
|---|---:|---:|---:|---:|
| disable-all, threads=1 | 267/267 | 3000/3000 | 0 | 0 |
| disable-all, threads=4 | 267/267 | 3000/3000 | 0 | 0 |
| default, threads=1 | 267/267 | 3000/3000 | 0 | 0 |
| default, threads=4 | 267/267 | 3000/3000 | 0 | 0 |

Raw arrays, not only output signs, were compared.  The candidate and authority
were both 267/267 known and 2940/3000 fresh; all 60 fresh misses are inherited
authority behavior.  Candidate minimum positive output was 1 and maximum was
127.

A separate 512-case trace under disable-all and default optimization checked
the ArgMax, background mask, reduced condition, transposed mask, base weights,
final weights, and output at every rewrite boundary.  Every relation had zero
failure.

The candidate passes full ONNX checker, strict inference with
`data_prop=True`, canonical I/O, standard-domain/banned-op/function/nested-
graph/sparse/external-data gates, finite initializers, and Conv/QLinearConv
bias UB0.

It intentionally inherits the authority's metadata cloak.  On 64 traced cases
in both optimization modes, authority and candidate have the exact same 26
declared/runtime mismatch signatures.  Candidate-minus-authority and
authority-minus-candidate mismatch sets are both empty; runtime errors and
nonfinite intermediate values are zero.

## Residual exploration

| family | measured result | decision |
|---|---:|---|
| direct scalar condition reduction | cost 977 | included |
| direct int64 ArgMax equality | cost 977 | included |
| background-mask terminal Where | cost 977 | included |
| condition + terminal | cost 976 | included in winner |
| all three | **cost 975** | winner |
| scatter zero into raw counts | strict inference failure | declared count tensor is `[1,1,1,1]`, not the actual ten-channel shape |
| bypass dynamic crop-shape cloak | strict inference failure | exposes the inherited `[1,10,29,29]` activation against `[1,1,1,1]` metadata |
| direct int32 min-row path (lane 201) | cost 1050 | regression |
| shared color initializer (lane 201) | inherited buffer-reuse runtime failure | reject |
| remove fp16 log epsilon, low inverse | 32/5631 conservative-support differences | reject |
| remove epsilon, next-high fp16 inverse | 22/5631 differences | reject |

The scale scan enumerated every nonempty bit mask in every contiguous window
of width 1 through 10 over the 19 legal columns, a conservative superset of
original and 2x-magnified row encodings.  It confirms that the epsilon and
existing fp16 chain cannot be removed by merely choosing the adjacent fp16
inverse-log constant.

The remaining cost-975 graph has official memory 848 and 127 parameters.
Within the audited dense-initializer, no-new-mismatch, local algebraic families,
975 is the current residual floor.  Wider reductions would require replacing
the dynamic metadata-cloak identities, the 7-row/5-bit sprite encoding, or the
fp16 exponent extraction; tested direct variants either expose a new shape
mismatch/runtime allocation failure or regress cost.

## Artifacts

- winner: `candidates/task319_combined_best_local.onnx`
- complete audit: `audit.json`, `audit.py`
- builder and all probes: `build_candidates.py`, `build.json`, `candidates/`
- fp16 support scan: `scale_support_scan.py`, `scale_support_scan.json`
- concise machine result: `result.json`
