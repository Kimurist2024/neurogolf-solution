# B27 task382 shape-honesty audit

Baseline score label: `8000.46`  
Baseline ZIP SHA-256: `74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534`

## Outcome

No candidate is eligible. The old cost-814 candidate is `REJECT_COST_AND_SHAPE`.

Changing only the graph output from `[1,10,1,1]` to the real `[1,10,30,30]` is not a valid repair:

- full ONNX checker rejects the final Einsum because its spoofed intermediate declarations still infer `[1,10,1,1]`;
- ORT with optimizations disabled remains correct on `266/266` known cases;
- default ORT fails on `266/266` known cases with allocator shape-reuse errors;
- Conv-family bias UB count is zero.

Updating all intermediate declarations to the observed full runtime shapes removes the allocator error and passes known cases in both ORT modes (`266/266`, errors `0`). It also exposes the actual cost:

- old claimed candidate cost: `814`;
- baseline task382 cost: `820`;
- observed-shape candidate cost: `55,417` (`55,275` memory + `142` params);
- score delta versus baseline: `-4.213337344611196`.

Additionally, gravity changes the row/column orientation. Eight intermediates swap between shapes such as `[1,2,30,1]` and `[1,2,1,30]`. A declaration copied from the horizontal runtime has eight contradictions on a vertical runtime, and vice versa. Therefore a static truthful repair requires an architectural rewrite, not a metadata edit.

The same-or-cheaper prerequisite failed decisively, so fresh 5,000-case and external acceptance validation were not run. This follows the requested early-rejection rule and avoids promoting an error-prone model.

Evidence is recorded in `audit.json`, `build_manifest.json`, and `winner_manifest.json`. task328 remains terminally rejected for margin failure.
