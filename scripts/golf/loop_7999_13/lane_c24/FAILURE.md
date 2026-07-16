# C24 no-promotion note

- task363: exact declared cost 513 is confirmed, but the model has seven
  runtime-shape mismatches and fails 21/5,000 legal fresh cases. The pure
  generator rule is non-identifiable against two inconsistent stored fixtures,
  so no deterministic true-rule model can satisfy every required case.
- task388: exact declared cost 311 generalizes on 5,000 fresh cases but has 14
  runtime-shape mismatches. The only historical model cheaper than 311 errors
  on all 266 known cases in both ORT modes. The cheapest fully shape-truthful
  known-perfect control costs 6,468.

No unsafe or weak candidate was promoted, and the root artifacts were not
modified.
