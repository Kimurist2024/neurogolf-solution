# 8004.42 improvement loop

## Immutable LB baseline

- Score: `8004.42` (LB verified)
- ZIP: `submission_base_8004.42.zip`
- MD5: `47442e8b6e778824c813a47230031e13`
- SHA-256: `14e14ab9844d150b582d87c050043184f40ea1d3baae9cda8000248b1ab117d0`
- Archive members: 400; integrity check passed; Conv-family bias UB count: 0

## Fixed LB improvements

These 18 members are the exact changes from the 8003.40 baseline and must not be
reverted while improving from 8004.42:

`15, 20, 31, 68, 71, 79, 88, 105, 174, 183, 189, 206, 221, 240, 243, 259, 300, 302`

## Safe prior improvements rebased onto 8004.42

The prior Wave 3 candidate was compared again against the 8004.42 ZIP. Nine
non-conflicting candidates remain cheaper and retain their previous validation
evidence:

`13, 109, 132, 158, 344, 349, 358, 379, 398`

Task 105 is deliberately not taken from the old Wave 3 ZIP: its old candidate
cost is 194, whereas the LB-fixed 8004.42 member costs 188.

- Cost reduction from the nine rebased members: 50
- Projected gain over the LB baseline: `+0.11261545568566689`
- Projected aggregate score: `8004.532615455686`
- Fixed ZIP: `submission_8004.42_fixed_rebase_meta.zip`
- Fixed ZIP SHA-256: `dc64ef5ac672fd7cc67418318e70f94cafd5396af8e25ad2939cdcd4eb1e0149`
- Build audit: `fixed_rebase_build_audit.json`
- Fresh differential audit: `fixed_rebase_compare500.json`

The metadata-safe build changed only the nine listed payloads. Member order,
per-member metadata, archive comment, and every unchanged payload are identical
to `submission_base_8004.42.zip`. ZIP integrity passed and the complete archive
has zero Conv-family bias-length UB findings.

Fresh differential results are exact for tasks 13, 109, 158, 349, 358, and
379. Tasks 132 and 398 are threshold-equal on 499/500 and 498/500 cases,
respectively, above the user-authorized 95% gate. Task 344 remains the previously
authorized local-rule candidate: known 266/266 and independent generator-fresh
4972/5000 (99.44%) under both ORT modes, with zero generated runtime errors.
Arbitrary-input comparison to the prior task344 implementation is intentionally
not treated as its truth oracle.

The projected score is not labeled LB verified until the resulting ZIP is
submitted and scored.

## Additional 20-task expansion

The requested twenty extra targets were frozen into two disjoint batches and
audited without modifying the fixed ZIP.

- Low batch: `25, 62, 8, 134, 112, 184, 168, 48, 37, 14`
- High batch: `374, 250, 324, 308, 275, 338, 333, 268, 377, 279`
- Safe adoptees: 0
- Additional projected gain: `+0.0`

Low-batch evidence is under `agent_batch20_low/`; high-batch evidence is under
`agent_batch20_high/`. Numerically cheaper candidates were rejected when they
used lookup/shape-cloak/giant floating contractions, failed strict data-prop,
fell below 95% fresh accuracy, or produced runtime errors. In particular,
task333's 423 -> 421 candidate was not admitted because it changes a 36-input
floating Einsum, and task377's 409 -> 408 candidate failed all known cases under
default ORT with runtime errors.

## Isolated relaxed-95 lead

Task 205 has an isolated `RELAXED95_CANDIDATE_ONLY` artifact under
`agent_archive_safe/`: cost 1042 -> 937 and projected gain
`+0.10621394007489116`, with known 266/266 and dual-ORT fresh 4904/5000
(98.08%), runtime errors 0. It is not in the fixed ZIP because its fresh result
is 24 cases worse than the incumbent and it contains a 13-input float32 Einsum.
