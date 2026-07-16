# plus20 target-mid17 report

## Outcome

No candidate is admissible for tasks `324`, `338`, `268`, or `184`.
Accepted count is **0**, projected gain is **+0.0**, and no ZIP was built.
`submission.zip`, `best_score.json`, `all_scores.csv`, `a.csv`,
`artifacts/handcrafted`, and both baseline archives were not modified.

The requested authority was `submission_base_8004.50.zip` (SHA-256
`63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`).
The later `submission_base_8005.16.zip` has SHA-256
`73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`.
All four member hashes are identical between those archives, so this result is
**rebase-compatible with 8005.16**.

## Raw rules and independent validation

- task324 / `d07ae81c`: identify the two background colors and the two seed
  colors, then extend both diagonals through every seed, preserving which
  background class each painted cell replaces.
- task338 / `d5d6de2d`: paint green exactly on black cells enclosed by red
  rectangle walls.
- task268 / `aba27056`: infer the unique open side of the hollow colored box,
  canonicalize its orientation, and paint the yellow interior, vertical stream,
  and two diagonal rays.
- task184 / `780d0b14`: split at all-zero separator rows/columns and recover the
  unique nonzero patch color in every 2x2--3x3 block grid.

The independent Python decoders match all raw known examples: task324
`266/266`, task338 `267/267`, task268 `266/266`, task184 `266/266`.  Each also
matches two independent fresh runs of `5000/5000`.  task184 has only 169 known
examples compatible with the fixed 30x30 ONNX carrier; the raw decoder still
checks all 266, and the two fresh sets contain 3153 and 3170 carrier-compatible
instances.  task324's upstream generator can loop forever when it draws no
stripes; its fresh sampler rejects that non-returning parameter case and samples
the same distribution conditional on a returned example.

## Strict disposition

| task | base cost | decisive base defect | cheapest relevant lead/control | decision |
|---:|---:|---|---|---|
| 324 | 439 | default ORT TopK failure; 5 false shapes; four giant Einsums, max 65 inputs | compliant rule control 16550 | no cheaper admissible model |
| 338 | 426 | default output-shape failure; runtime allocator mismatch | cost424 errors 267/267; cost740 exact but 29-input giant Einsum; compliant rule control 37101 | no cheaper admissible model |
| 268 | 422 | 2 TfIdf lookups, 31 CenterCropPad nodes, 32 false shapes, default ORT failure | cost327 is known-complete only under DISABLE_ALL but fresh 43.74% / 42.82%; compliant rule control 18665 | reject lookup/private lineage/fresh/shape/default |
| 184 | 421 | 6 false shapes, default ORT failure, true one-case footprint 64961 bytes | closest complete-known history 422; compliant rule control 1996 | no strict decrease |

The compliant rule controls pass full ONNX checking, strict shape inference
with data propagation, truthful runtime-shape tracing, standard-domain checks,
both ORT modes with zero known errors, and UB0.  Their actual costs are far
above the immutable members.  Conversely, every apparent sub-baseline lead
depends on a forbidden shape cloak/lookup/giant contraction, fails default ORT,
fails known execution, or fails the private-lineage 100% fresh requirement.
There is no all-input equivalence proof to a clean lower-bound member that would
permit an exception.

## Evidence

- `STRUCTURAL_AUDIT.json`: exact fixed members and the closest historical leads.
- `CONTROL_AUDIT.json`: truthful rule controls and the explicitly rejected
  giant-Einsum/control variants.
- `REFERENCE_AUDIT.json`: known and two-seed 5000x2 rule validation.
- `CANDIDATE_FRESH.json`: independent two-seed rejection of task268 cost327.
- `RESULT.json`: machine-readable final disposition.
- `audit_targets.py`: reproducible, non-promoting audit driver.
