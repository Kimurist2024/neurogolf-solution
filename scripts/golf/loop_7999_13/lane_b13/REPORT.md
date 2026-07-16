# Lane B13 — task254/task267 safe rebuild

## Result

No candidate is eligible for adoption. The exact `submission_base_7999.13.zip`
remains unchanged; lane score delta is `0.0`.

| task | exact base | best safe result | decision |
|---:|---:|---:|---|
| 254 | 76 | no correct model below 76 with at most 16 Einsum operands | reject all |
| 267 | 60 | safe zero-rebuild control cost 60 (tie) | no adoption |

## Task254

The generator rule is fixed 9x9 gray bars: the shortest bar becomes red and the
tallest becomes blue. The exact base is cost 76 but contains Einsums with up to
49 operands, so it is not a B13-safe construction.

Three independent routes were checked:

1. Constant precontraction produced a cost-64, 15-operand model, but eliminating
   the coupled `e/f` latent state changed the function. It scored 0/265 known in
   both ORT modes and was immediately rejected.
2. A four-core tensor-train keeps the single graph node at exactly 16 operands.
   60 candidates spanning cost/params
   58–74
   were tested. None solved any complete known case; runtime errors were zero.
   Approximation perturbed required zero logits across the threshold.
3. Exhaustive exact-rank analysis over all six feature-axis orders and all
   four-core cuts found an exact-family floor of
   114 params, above cost 76.

The correct archive cost-68 witness still needs 20 Einsum operands and is
therefore rejected by the explicit >16 rule.

## Task267

The generator rule is fixed 7x7 creature recoloring from the marker at `(6,0)`.
The from-scratch standard control uses one 5-input Einsum, finite initializers,
no intermediates, and cost 60. It passed all 264 known cases in both ORT modes
with zero runtime errors, but ties the exact base and cannot improve the score.

The archive cost-30 model remains ineligible: it uses a 37-input giant Einsum
and numerical repeated-product behavior. No lookup, UB, shape cloak, sparse
initializer, custom domain, or nonstandard operator was adopted.

## Gates and files

- `audit.json`: checker, strict shape inference with data propagation, domains,
  finite initializers, Conv bias, operand counts, actual costs, and both-ORT known results.
- `tt_search.json`: all 60 under-budget TT attempts and their complete-known results.
- `winner_manifest.json`: empty winner list.

Fresh 5000 x both ORT modes was intentionally not run because no candidate
passed the required complete-known pre-gate. Root ZIP/CSV/ledger files were not modified.
