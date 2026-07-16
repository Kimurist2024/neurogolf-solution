# task101 / task133 exact-regolf lane (8009.46 authority)

## Outcome

One new target file survives the lane's fail-closed gates:

| task | immutable base | candidate | gain | decision |
|---:|---:|---:|---:|:---|
| 101 | 5655 | 5641 | +0.002478754810 | exact-regolf winner |
| 133 | 4393 | — | +0.000000000000 | no safe lower candidate |

The winner is `winner/task101.onnx`, SHA-256
`a57a944d958be1945563a7d55320239bb0f36b4ba25af1041f589a904cc7b81e`.
It was not merged into a submission ZIP, score CSV, or root model.

## task101 proof and validation

The immutable `submission_base_8009.46.zip` member contains
`And(tail3x6, is_wide_bool)`, where `tail3x6` is an all-true Boolean tensor of
shape `[1,1,3,6]` and `is_wide_bool` is a scalar Boolean tensor of shape
`[1,1,1,1]`. The rewrite is:

```text
And(all_true[1,1,3,6], b) -> Expand(b, [1,1,3,6])
```

This preserves both every Boolean value and the exact output shape. It removes
the 18-element initializer and adds a 4-element shape initializer, reducing
parameters from 114 to 100 while memory remains 5541. Actual official-like
cost is therefore `5655 -> 5641`.

Validation results:

- full ONNX checker: pass;
- strict shape inference with data propagation: pass;
- runtime/declaration mismatches: 0;
- known corpus: 266/266, errors 0, in both `ORT_DISABLE_ALL` and default ORT;
- disjoint fresh range 1011181: 4950/5000 (99.00%), errors/nonfinite 0, both ORT modes;
- disjoint fresh range 2011181: 4949/5000 (98.98%), errors/nonfinite 0, both ORT modes;
- candidate vs immutable 8009.46 raw equality: 5000/5000 per seed and ORT mode, maximum raw delta 0;
- candidate disabled/default raw equality: 5000/5000 per seed, maximum raw delta 0;
- standard domain only, no function/sparse initializer/banned op/lookup flag/giant Einsum;
- project Conv-bias checker findings: `[]`.

The current task101 lineage is not a generator-perfect true-rule model; the
fresh failures are shared exactly by the accepted 8009.46 member. Admission is
therefore the user's private-zero exception: this is an exact no-regression
regolf of the already accepted payload, not a claim that the payload implements
the generator perfectly. The pre-existing dynamic QLinearConv bias subgraphs
are unchanged by this rewrite; the project checker reports no finding and both
ORT modes are bit-identical on all 10,000 independent fresh cases.

## task133 stopping result

The exact member is unchanged from the previously audited SHA
`6c5dc3a593b0900e16966b9d4c40af509a34c1dd1f0264c31cd30eaf9b4570e5`.
Mechanical dead/initializer-alias/no-op/CSE/constant-fold/factor-absorb scans
find no semantic reduction. It retains 30 runtime/declaration shape
contradictions.

Removing all 238 stale `value_info` annotations creates SHA
`f7dd6c37e74f1d6e6cc88b2f3311fb1ce667a7fc62717f7eac4c75d53bedf24a`,
but that graph is not fully statically shaped and is not scorer-profileable, so
it is rejected. The independent truthful generator-rule control remains cost
5570, 1177 above the incumbent. There is no task133 winner.

## Rejected task101 variants

- Generic `And` bypass replaced a broadcast tensor with a scalar and inferred
  height 15 instead of 17. Checker and ORT reject it.
- Boolean `Resize` would save two more parameters in theory, but ORT 1.24 has
  no implementation for `Resize(bool)` and rejects session creation.

Only `winner_manifest.json` is authoritative for integration. Files under
`candidates/` are audit evidence and must not be merged directly.
