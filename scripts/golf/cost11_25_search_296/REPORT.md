# Cost 11–25 strict-lower search (lane 296)

## Result

No candidate in this defined search survived the exact-known admission gate.
Consequently, no candidate was promoted, no fresh/private-zero inference was made,
and no root or staging artifact was changed.

This is a negative result for the families and history enumerated below, not a
proof that no smaller formula can exist.

## Authority and scope

- LB authority: `submission_base_8011.05.zip` (`8011.05`)
- SHA-256: `ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56`
- Scope: every authority member with audited cost 11 through 25 inclusive
- Tasks (27): `43, 45, 47, 52, 56, 63, 73, 82, 103, 110, 116, 130, 150, 155, 164, 166, 172, 186, 210, 293, 299, 311, 314, 322, 359, 372, 399`
- Admission policy: strictly lower actual cost, accepted structure, exact known
  behavior, four ORT configurations, then truthful static-shape/margin checks and
  fresh `2000 x 2` only for an exact-known finalist.

## Main enumeration

`scan.py` evaluated 3,783 candidate-task pairs. It covered score-25-style
zero/one-parameter operators, unary/binary/spatial operators, reductions and
pooling, legacy-compatible operator forms, task-aware initializer/optional-input
reductions, rank-one latent-axis reductions, and every lower-cost entry for these
tasks in `lane_archive_all400/inventory.json`.

Classification:

- `REJECT_QUICK_KNOWN`: 3,663
- `REJECT_STRUCTURE`: 111
- `REJECT_NOT_STRICT_LOWER`: 9
- admitted finalists: 0

The closest quick-screen results were still non-exact:

- task 56: historical cost 18 model, 10/12
- task 103: Conv bias removal, cost 14, 7/12
- task 45: identity, cost 0, 4/12
- every other task: best 0/12

Because none passed the 12-case exact pre-screen, running fresh validation would
not have made any candidate admissible.

## Focused follow-ups

- Finite ConvTranspose reconstruction: 7,464 cropped/rebased candidates over
  tasks 164, 172, 210, 311, 322, 116, and 372. Every best result was 0/12;
  there were no exact candidates.
- ConvTranspose bias-removal feasibility:
  - task 314: the complete known sign-constraint linear program is infeasible.
  - task 322: seven identical-feature/opposite-label conflicts prove that the
    bias-free form cannot fit the complete known set.
- ConvInteger optional-input removal failed on the complete known sets for tasks
  56, 186, and 399.
- Rank-two-to-rank-one Einsum reductions, input-only sub-equations, task 82 kernel
  support crops, task 186 neutral-padding crops, and task 399 unscaled power-sum
  encodings did not produce an exact known solution.
- ReverseSequence is not admissible: the official structural policy rejects op
  types containing `Sequence`, and ORT 1.24 also restricts `time_axis` to axes
  below 2, so it cannot implement the required spatial flip.
- Sparse-initializer and non-finite/shape-cloak candidates were excluded under
  the requested strict safety policy.

## Evidence

- `evidence.json`: full 3,783-row main enumeration and classifications
- `finite_conv_scan.json`: 7,464 finite ConvTranspose reconstruction attempts
- `lp_conv_no_bias.json`: exact LP/conflict certificates for tasks 314 and 322
- `scan.py`, `finite_conv_scan.py`, `lp_conv_no_bias.py`: reproducible search code

## Root-integrity check

At report time, `submission.zip` and the authority archive had the same SHA-256:
`ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56`.
The search wrote only within this evidence directory.
