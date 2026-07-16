# Cost 2–50 half-cost lane (8011.05 authority)

## Result

- Scope: all 79 non-score-25 authority tasks with cost 2 through 50.
- Guaranteed-safe strict-lower improvements: **0**.
- POLICY95 improvements: **task202**, with one preferred non-giant cost-20
  candidate and two retained fallbacks.
- Root submission, `all_scores.csv`, and staging directories were not changed.

## Coverage

This lane reused the complete fail-closed evidence from lanes 295/296/297/298/301
and added task202 algebraic operand ablations and independent fresh validation.
The reused evidence contains at least **25,410 evaluated candidate rows**:

- cost <= 10 second pass: 7,489
- cost 11–25 general enumeration: 3,783
- cost 11–25 finite ConvTranspose reconstruction: 7,464
- cost 26–50 generic/native enumeration: 6,337
- cost 26–50 latent pruning: 44
- task188 focused scale-free attempts: 218
- task399 focused micro-cut: 1
- retained-history strict-lower rows: 74

This lane added 32 focused task202 evaluation rows (operand ablations, parameter
sharing, retained variants, and full four-ORT audits), for a conservative total
of **at least 25,442 candidate evaluations** across the 79-task scope.

## Preferred POLICY95 finalist

`task202_policy95_arity14_cost20.onnx`

- authority cost: 48
- candidate cost: 20 (48 -> 20, reduction 28; half-cost target met)
- projected score gain: `ln(48/20) = +0.8754687374`
- full checker, strict shape/data propagation, canonical I/O, truthful runtime
  shape trace: pass
- banned/Sequence/lookup/sparse/functions/nested graphs/external data: none
- nonfinite initializers/outputs, Conv bias UB, runtime errors, shape cloak: none
- Einsum operands: 16 -> 14, below the giant threshold
- complete convertible known set: 230/230 exact in all four ORT settings
- independent fresh seed `304202014`: 1,948/2,000 = **97.40%** in each
  of four ORT settings
- independent fresh seed `304202114`: 1,933/2,000 = **96.65%** in each
  of four ORT settings
- combined independent fresh: 3,881/4,000 = **97.025%**
- cross-setting sign differences: 0
- runtime errors/nonfinite cases/nonfinite elements/shape mismatches/small-positive
  elements: 0 for both seeds and all four settings
- minimum positive raw value across both seeds: `1.5169620513916016`;
  maximum nonpositive raw value: `0.0`

The two removed operands are positive active-width and active-height factors.
Removing them divides relevant raw outputs by a positive factor and therefore
does not change the sign threshold; it also removes the only structural objection
attached to the retained 16-input lineage.

Evidence: `task202_arity14_evidence.json`

## Retained fallbacks

1. `task202_fallback_cost28.onnx`: cost 28, known 230/230, fresh
   1,996/2,000 = **99.80%** in all four settings. It is output-only but has a
   21-input Einsum, so it requires the explicit giant-Einsum exception and does
   not meet the half-cost target. Projected score gain: `+0.5389965007`.
2. `task202_policy95_cost20.onnx`: original finite cost-20 lineage, known
   230/230, independent fresh seeds each 1,931/2,000 = **96.55%** in all four
   settings. It has a 16-input Einsum and is superseded by the preferred arity-14
   candidate. Its two-seed evidence is preserved in
   `task202_policy95_evidence.json`.

The color-symmetric cost-20 experiment reached 98.55% fresh but only 225/230 on
the complete known set, so it was rejected and is not a finalist.

## No guaranteed-safe adoptee

The only other lower-cost known-exact history in this range was task322/task372;
both use malformed ConvTranspose bias/nonfinite behavior and remain rejected.
No other candidate in the enumerated finite, native-op, constant, broadcast,
initializer-pruning, latent-pruning, transpose/pad/slice, or history families
survived strict-lower + complete-known + structural admission.
