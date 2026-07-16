# task192 exact shared-basis factorization 178

## Result

Accepted SOUND stage candidate:
`candidates/task192_shared_basis_argmax.onnx`, SHA-256
`19fbdce89a5c89f5ff376b2fbbdb630ead5535d5ed5ebe7d9914a4de89e5023c`.

- immutable8009.46 authority: cost1609;
- prior staged exact candidate: cost1195;
- new candidate: memory208 + params941 = **1149**;
- authority-relative projected gain: **+0.336720869144**;
- incremental gain over the prior stage: **+0.039254186517**.

`others/71407/task192.onnx` was replaced with this exact SHA. The root
submission, baseline ZIP, and score ledger remain unchanged.

## Exact rewrite

The prior polynomial separately materialized `[all, selected]` and
`[background, selected]` tensors. The new graph constructs one shared basis:

```text
basis = [nonzero, background, selected]
```

Three integer 2x3 matrices recover the original factors inside the final
Einsum:

```text
center   = [inside, nonzero]
neighbor = [inside, selected]
route    = [background, -9*background + selected]
```

Thus the final expression is unchanged:

```text
B * background + P * (-9*background + selected)
```

where `B` is the inside-grid horizontal/vertical product and `P` is the
nonzero-center, selected-horizontal, selected-vertical product. The histogram
uses the explicit nonzero mask. Selection retains the accepted
ArgMax(first-tie)+OneHot implementation.

All factors and valid-grid contractions use small integers exactly
representable in float32. The previous exhaustive163-case sign argument
therefore applies without a rounding-margin assumption.

## Verification

- official-like profile: memory208, params941, cost1149;
- ONNX full checker and strict data-propagating inference: pass;
- static and runtime output shapes truthful: pass;
- standard domain, Conv-family UB0, no sparse/lookup/Hardmax: pass;
- known265/265 in disable-all threads1/4 and default threads1/4;
- fresh seed192800661: 5000/5000;
- fresh seed192930007: 5000/5000;
- runtime errors and nonfinite outputs: 0;
- exhaustive local sign cases: 163/163.

The exploratory Hardmax variant costs1138 and also passed known/fresh, but the
SOUND structural gate classifies Hardmax as lookup. It is rejected and was not
staged.

Machine-readable evidence is in `build_safe_argmax.json` and
`audit/task192_exact_poly.json`.
