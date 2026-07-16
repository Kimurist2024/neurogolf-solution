# Lane A32 — task288 / task335 strict optimization

## Result

**No lower-cost candidate is retained; lane gain is 0.**  Wave16 and all
protected root artifacts remain unchanged.

## task288

Wave16 member:

- SHA-256: `44dd1d5a6dc7e1ee005d0ea94e1ff53707c9995ceb5a96c396446478bf12f37a`
- official-like cost: memory `40` + parameters `75` = **115**
- known: **267/267**, errors 0
- fresh generator: **5000/5000** under `ORT_DISABLE_ALL` and **5000/5000**
  under default ORT, generation/runtime/output failures 0
- full checker and strict inference expose the only intermediate truthfully as
  `bias4 [1,10,1,1]`; public I/O is `[1,10,30,30]`

The graph is the existing spec-derived two-ConvTranspose rule implementation.
Its counted support is 55 render taps, 10 probe taps, and a schema-required
10-channel probe bias.  A generator-complete LP over all six legal
`neck x shoulder` geometries established:

- fixing both probe biases to zero is infeasible;
- forcing the background/non-background biases equal is infeasible;
- deleting either render edge row, either render edge column, either probe edge
  row, or either probe edge column is infeasible (tested costs 54--63).

The initializer tensors contain no zero border or duplicate row/column that can
be cropped.  Shortening the 10-channel ConvTranspose bias would introduce the
explicitly prohibited bias-length UB.  No lower candidate was emitted.

## task335

Wave16 member:

- SHA-256: `79da8462ed32fe2ea46677637f51923cd6e4abc31fe94e7b816e3599aeba0d57`
- cost: memory `0` + parameters `109` = **109**
- known: **266/266** in both ORT modes, errors 0
- fresh control: **4970/5000** in both ORT modes; the same 30 generated cases
  are output mismatches, so the incumbent itself is not a strict 100% fresh
  reference

The three `S=[0,0,-1]` state contractions can be coupled exactly.  The resulting
`task335_coupled_s.onnx` reduces the existing Einsum from 44 to 42 operands,
but parameters, memory, and cost remain 109.  Independent validation records:

- known 266/266, errors 0;
- generic differential **500/500 raw-bitwise equal**, threshold mismatches 0,
  asymmetric/runtime failures 0, maximum raw difference 0;
- verdict `REJECT` because official-like cost does not decrease.

All actual cost-reduction probes failed before fresh promotion:

| candidate family | cost/params | disabled ORT | default ORT | decision |
|---|---:|---:|---:|---|
| coupled state + C row-sum selector | 106 | 0/266 | 0/266 | reject |
| coupled state + scalar S | 107 | 0/266 | 0/266 | reject |
| remove S plus one color test | 106 | at most 1/266 | at most 1/266 | reject |
| share B or M01 | 101 | 0/266 | 0/266 | reject |

`B` and `M01` are full rank in every mode and are not permutation/scale copies.
`C`'s public color axis cannot be shortened truthfully.  No new or enlarged
Einsum, lookup table, sparse initializer, or shape cloak was introduced.

## Structural safety

Full ONNX checker and strict inference pass for every emitted probe.  The
Wave16 archive-wide Conv bias-length audit reports **0 UB tasks**.  No probe
passed the known-complete and lower-real-cost gates, so no candidate ZIP or
fresh promotion artifact was produced.

Evidence: `candidate_screen.json`, `baseline_fresh5000.json`,
`task288_external_known.json`, and `task335_coupled_external500.json`.
