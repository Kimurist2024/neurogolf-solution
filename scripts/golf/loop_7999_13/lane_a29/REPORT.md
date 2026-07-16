# Lane A29 — task275 / task308 strict optimization

## Result

One strict winner is retained: **task275 cost 432 -> 428**, projected score
gain **+0.00930239266231341**.

- Candidate: `task275_shared_gate_router.onnx`
- Candidate SHA-256: `d31e860f7243a66917a06b306b6bd856da71a0f09dfc4d4d0e6472f1c9e2003f`
- Exact Wave14/base member SHA-256:
  `0223b550b6555a701e44a2d5b6a77afdf4650f2225f398c5d9096e65869e8d11`
- Cost: memory `12 -> 12`, parameters `420 -> 416`, total `432 -> 428`

No root submission ZIP, CSV, score pointer, shared handcrafted artifact, or
protected baseline was modified.

## Why task275 is generator-exact

The initializer rows satisfy `GV[0] = -GU[0]` and `GV[1] = 7*GU[1]`.
Consequently, in real arithmetic, for every `total`:

`(total-25) GU[0] x GV[0] + GU[1] x GV[1]`

equals

`(25-total) GU[0] x GU[0] + 7 GU[1] x GU[1]`.

The generator has only `size=3` and `size=4`.  Its input contains respectively
`3 x 6 = 18` or `4 x 8 = 32` one-hot cells, so its only reachable gate totals
are 18 and 32.

The incumbent forms a 2x2 router from `gate=(total-25, 1)`, `GU`, and `GV`:

- total 18: `[[14, 0], [0, 14]]`
- total 32: `[[0, 14], [14, 0]]`

The candidate changes the gate to `(25-total, 7)` and uses `GU` for both router
factors.  It produces those same two matrices exactly.  Therefore `GV` is no
longer needed and its four parameters are removed.  `winner_audit.json`
machine-checks both matrices with exact float32 equality.

This does not add or enlarge an Einsum.  Base and candidate both have exactly
three nodes (`ReduceSum`, `Conv`, `Einsum`), and the final Einsum retains the
same 41 operands.  The candidate only rewires four existing inputs from `GV`
to `GU` and changes the two existing Conv initializers.

## Mandatory validation

- Independent team validator: known **266/266**, errors **0**, cost **428**,
  `ACCEPT_STRICT`; baseline known 266/266, cost 432.
- Independent random differential: **100/100 threshold-identical**, runtime
  failures 0.  Raw magnitudes differ from reassociation, but signs do not.
- Fresh generator: **5000/5000** with `ORT_DISABLE_ALL` and **5000/5000** with
  default ORT; runtime/output failures **0** in both modes.
- Dual-ORT known audit: candidate 266/266 in each mode, wrong 0, errors 0,
  non-finite values 0.
- Margin: minimum positive raw value `38415.98046875`; maximum non-positive
  raw value `0.0`.
- Full ONNX checker and strict shape inference: pass.
- Truthful input/intermediate/output shapes:
  `[1,10,30,30] -> [1,1,1,1], [1,2,1,1] -> [1,10,30,30]`.
- Functions, sparse initializers, nested graphs, foreign domains, banned ops,
  ConvTranspose, and QLinearConv: all absent.

Evidence: `winner_audit.json`, `task275_shared_gate_router_fresh5000.json`,
`task275_shared_gate_router_external100.json`.

The team validator's broader 500-case generic fuzz set was also retained as
`task275_shared_gate_router_external500.json`: it is 494/500
threshold-identical with no runtime failures.  Its six differences are
floating-point reassociation at invalid off-generator layouts (the first is a
2x9 grid; this task can only generate 3x6, 6x3, 4x8, or 8x4).  It is not used
as generator correctness evidence.  The official generator domain is covered
by the dual-ORT 5000-case audit above, with a minimum positive margin over
38,000.

## task308 outcome

No winner is retained; exact Wave14 cost remains **434**.

- Reusing `idx30` as the dtype-only `CastLike` anchor is raw-bitwise identical
  on executable differential cases, but memory/params/cost remain
  `376/58/434`, so it is rejected.
- Reusing the earlier rank `Shape` output for `TopK` removes one node on paper,
  but ORT rejects the model at session creation with
  `TopK [ShapeInferenceError] Axis has less than the requested k elements`.
- The remaining initializers have schema-required distinct dtype/rank roles.
  `four_f` is the coefficient in both variance contractions; `zero_u8_4` is the
  output-base zero; the 30-coordinate vector and 4x4 gather map cannot be
  shortened without adding a charged reconstruction or invalidating the
  30-axis contractions.

Evidence: `task308_castlike_idx_anchor_external20.json` and
`task308_reuse_rank_shape_external100.json`.

## Other rejected probes

Four direct/transpose variants that shared task275's unrelated latent color
maps `T` and `W` each reduced nine parameters but scored 0/266 known-correct.
They were terminally rejected before fresh validation.  Their external screen
records are the four `task275_share_*_screen.json` files.
