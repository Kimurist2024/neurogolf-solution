# 8012.15 cost<=166 optimization — 3-worker low-cost transfer

## Result

The immutable authority is `submission_base_8012.15.zip` (LB 8012.15,
SHA-256 `1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231`).
All 222 non-score25 tasks at cost at most 166 are covered. No member in this
scope changed between the 8011.05 and 8012.15 authorities.

- Guaranteed-safe new candidates: **0**.
- Admitted under the user's 90% policy: **task175, cost 166 -> 145**.
- Conditional score gain: **+0.1352540459**.
- Conditional projected LB: **8012.2852540459**.
- Admitted half-cost candidates: **0**.
- The root champion and `others/` were not changed.

## task175 admission

`candidates/task175_policy95_cost145.onnx` was re-profiled directly against the
current authority and costs 145 (memory 0, params 145). It is not in the known
leaderboard-black catalog.

| gate | result |
|---|---:|
| Known corpus, DISABLE_ALL threads 1/4 | 262/266 = 98.4962% in each |
| Known corpus, default optimizer threads 1/4 | 262/266 = 98.4962% in each |
| Fresh seed 405200175, 2,000 cases, all 4 configs | 100% in each |
| Fresh seed 405300175, 2,000 cases, all 4 configs | 100% in each |
| Runtime errors / nonfinite / shape mismatch | 0 / 0 / 0 |
| Small positives `(0,0.25)` / config sign mismatch | 0 / 0 |
| Static shapes / finite initializers / UB / shape cloak | pass / pass / none / none |

This is a `POLICY90` admission, not a guaranteed-safe result, because four
known cases do not match. Full evidence is in
`task175_policy95_rebase8012_audit.json`.

## Searches completed

Three execution workers ran concurrently:

1. History reprice: 10,188 loose ONNX files were enumerated and 594 theoretical
   strict-lower task/SHA candidates were fully gated. Twenty-one duplicate
   exact-lower rows survived known correctness, representing only previously
   known unsafe/quarantined lineages.
2. Cost<=10 structural transfer: all 21 reference tasks collapsed to 17 unique
   graphs; 13 finite/static templates were admitted and four nonfinite templates
   rejected. The 57 cost101..166 tasks received 8,502 candidate evaluations.
   No new strict-lower finalist survived.
3. Current graph contraction: 1,913 node-bypass, optional-operand, finite
   ConvTranspose-crop, and Einsum-factor ablations were evaluated across the
   same 57 tasks. No strict-lower finalist survived.

The earlier byte-identical cost11..100 search contributes another 22,280
cost<=10-pattern evaluations. Thus the new search did not simply repeat a small
visible-example probe; it covered retained history plus all low-cost structural
families available in the repository.

## Mandatory exclusions

- **task070**: cost52 is now confirmed leaderboard-black/error. Every task070
  reduction is excluded; cost50 additionally has forbidden small positives.
- **task202**: repeated private-zero lineage. Its apparent 48->20 half-cost win
  is excluded from checkpoint admission.
- **task271**: cost126 is known-perfect, but an independent dual-runtime fresh
  audit passed only 2/5,000; the lookup-net guarantee failed.
- **task391**: multiple lower lineages are independently leaderboard-black.
- **task322/task372**: cost19/12 lineages use nonfinite ConvTranspose data and
  unsafe short-bias behavior.

Machine-readable decisions are in `admission_decisions.json`; aggregate metadata
is in `MANIFEST.json`. No ZIP merge was performed by this lane.
