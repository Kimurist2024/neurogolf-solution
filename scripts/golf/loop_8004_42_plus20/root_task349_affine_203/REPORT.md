# task349 exact affine-table shave

## Outcome

One strict-lower exact-support descendant is ready for independent review. The
first cost3550 implementation and no-new-scalar cost3549 implementation were
both superseded before staging by the compound cost3548 implementation below.

- parent: `others/71407/task349.onnx`
- parent SHA-256: `179bbed5bd313a1f6ec62f573fd725ab71ff55a9509daaceff3f40274ac514c7`
- parent cost: `3229 memory + 327 params = 3556`
- candidate: `task349_affine_max29.onnx`
- candidate SHA-256: `f7531b66a5399973ed57835584023c5bf1f61966c218b283cb721ba7ca45c8e2`
- candidate cost: `3233 memory + 315 params = 3548`
- incremental gain over staged parent: `ln(3556/3548) = +0.002252253204`
- total gain over immutable task349 cost3564: `ln(3564/3548) = +0.004499445161`

The root archive, score ledger, and active staged ONNX were not changed by the
build or root audit.

## All-input identity

The parent has two 11-entry int8 lookup tables indexed by the same
`radius_code`. Exhaustive comparison of all 11 rows proves

`hstart_offset_by_mod_i8[i] = 1 - 3 * hend_offset_by_mod_i8[i]`.

The candidate removes the 11-element `hstart` table and adds no initializer.
For the remaining radius-table value `r`, it computes `twice=Add(r,r)`,
`top=Sub(1,twice)`, and `hstart=Sub(top,r)`. Thus `top=1-2r` and
`hstart=1-3r`. The radius table values are in `[0,5]`, so all intermediates and
the complete result range `[-14,1]` are representable in int8 without
overflow. This is an exact table identity for every possible index accepted by
the parent, not a sampled generator claim.

The added arithmetic has one more four-byte live output than the removed
`Gather+Add` pair. Parameters fall by eleven elements, for net cost -7.

## Generator-support max30 identity

The generator fixes `size=5*factor` with factor2..6, so every valid grid has
side in `{10,15,20,25,30}`. The one-hot scorer input sums to `side^2`, and the
graph obtains `side` by `Sqrt(ReduceSum(input))`. `halo_end` is explicitly
clipped to `[0,side]`; therefore both values tested by the parent's two
`Equal(x,30)` nodes are integer int8 values at most30. On that complete valid
support, `Equal(x,30)` is identical to `Greater(x,29)`. Reusing the existing
`max29_i8` initializer in both tests removes the now-dead `max30_i8` scalar
without changing memory, reducing cost3549->3548.

## Root audit

- full checker, strict inference, and strict data propagation: pass
- Conv-family nodes/bias-UB opportunities: zero
- runtime shape trace: 120 tensors, mismatch0, nonfinite0
- known267/267 in disabled/default ORT x threads1/4: candidate correct and
  raw-bitwise equal to parent in all four configurations
- fresh seeds20334911 and20334929, each2500 cases x four configurations:
  raw-bitwise equality5000/5000 per configuration, runtime errors0,
  candidate nonfinite0
- separate seed20334901, 5000 cases x disabled/default: raw equality10000/10000,
  runtime errors0
- `verify_fix.py --k 2000 --min-fresh-rate 0.9`: ADOPT, fresh1984/2000
  (99.2%), lib/official gold pass, margin stable with minimum2.0, cost3548

The parent and candidate share the same fresh truth results, while the exact
all-index identity guarantees pass-through on the parent's entire defined
domain. Evidence: `build_no_scalar.json`, `build_max30.json`, `audit.json`, and
`fresh_final_seed_20334901.json`. The older `build.json` and
`fresh_seed_20334901.json` describe only the superseded cost3550 diagnostic.

Independent lane204 repeated the cost/profile and mechanical graph diff,
proved the generator side bound from the source AST, traced the halo bound,
and passed known267x4 plus two new2500-case seeds x four configurations
(20,000/20,000 raw-bitwise comparisons), with error/nonfinite/mismatch0. The
final SHA was then staged as `others/71407/task349.onnx`. Evidence:
`agent_review_task349_affine_204/REPORT.md` and `audit_max29.json`.
