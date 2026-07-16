# C26 strict optimization report — tasks 310 and 328

## Outcome

No candidate is safe to adopt. The accepted list is empty and projected gain
is **+0.0**. No root submission, CSV, score ledger, best-score file, or shared
handcrafted artifact was modified.

Both exact 7999.13 members are unchanged in Wave12. Their SHA-256 values are
`f7ad4fb8...61d79` (task310, cost 566) and `08ba1aa5...f576a`
(task328, cost 558).

## Generator rules

- task310 (`task_c909285e.py`): generate a 20--30 square periodic wire grid,
  insert one complete square perimeter of side 5--8, and return precisely that
  square crop.
- task328 (`task_d22278a0.py`): place two to four distinct colors at sampled
  corners of a 6--18 square. For each output cell, paint the uniquely nearest
  corner color only when its Chebyshev distance from that corner is even;
  Manhattan-distance ties and odd Chebyshev distances remain zero.

## task310 decision

The exact cost-566 graph has no supplied `value_info`, passes full checker and
strict data-propagating shape inference, and introduces no custom domain,
banned op, nested graph, sparse initializer, or Conv-bias issue. Its single
large final Einsum has 23 operands and is inherited unchanged.

The previously constructed truthful selector
`lane_c9/task310_safe_linear_selector.onnx` is 5000/5000 with zero errors under
both default ORT and `ORT_DISABLE_ALL`, but costs 633, a score loss of
0.1118763439. It is therefore rejected. The 30-row `E3/E4` periodic factors
serve both input-coordinate and output-coordinate contractions; splitting off
their eight output rows or deriving the cap at runtime costs more than the
saved parameters. Existing history and all 48 individual optimizer passes
produce no cheaper truthful model.

A tempting one-comparison selector (`count < 29`) is unsound: a 50,000-case
generator audit found 22 legal cases with a non-frame wire count of 27. The
incumbent's exact `count < 25 OR count == 28` predicate cannot be collapsed to
that comparison over the generator domain.

Decision: **REJECT_NO_CHEAPER_STRICT_WINNER**.

## task328 candidate and rejection

The only new cheaper construction reuses the existing scalar
`ninvB = [-1/3]` as the first concatenated feature instead of storing
`one = [1]`, and multiplies `CoreI[:,0]` by -3. Float32 computes
`(-1/3) * (-3)` as exactly 1.0. This removes one parameter and changes no node,
runtime tensor shape, operator, or Einsum operand count:

| model | memory | params | cost | nominal gain |
|---|---:|---:|---:|---:|
| exact base | 200 | 358 | 558 | — |
| `task328_r01_reuse_ninv.onnx` | 200 | 357 | 557 | +0.001793722454 |

The external validator reports candidate and baseline both 267/267 known,
wrong 0, runtime errors 0, preflight warnings 0. Full checker, strict shape
inference, static-positive dimensions, standard domains, and no supplied
`value_info` all pass. The inherited final Einsum remains at 58 inputs; the
candidate does not enlarge it.

The strict numerical gate nevertheless rejects it:

- On four fresh cases in each ORT mode, decoded outputs match 4/4 and runtime
  errors are zero, but raw tensors are bitwise equal on 0/4. Maximum absolute
  raw difference is `1.2379400392853803e+27`, caused by changed multiplication
  order in the huge-coefficient contraction.
- On a deterministic 16-case dual-ORT probe, both base and candidate are
  correct 16/16 with zero errors, but each has three cases containing positive
  values in `(0, 0.25)`.

Thus the candidate is neither margin-stable for normal adoption nor raw
bitwise-equivalent for the user's >=95% exception. Running fresh5000 cannot
repair either already-observed terminal condition.

Historical cost-352--518 variants were also not adopted: the cost-427 model
already fails the margin gate (`3.73e-17` minimum positive), while the other
variants timed out in quick validation and/or increase the already giant
Einsum operand count. None supplies stronger strict evidence than the exact
base.

Decision: **REJECT_MARGIN_AND_NOT_RAW_EQUIVALENT**.

## Evidence

- `decision.json`
- `structural_audit.json`
- `task328_r01_external0.json`
- `task328_r01_exact4.json`
- `task328_r01_dual16.json`
- `task328_base_dual16.json`
- `task310_count_audit.json`
- `build_candidates.py`
- `parallel_fresh_task328.py`
