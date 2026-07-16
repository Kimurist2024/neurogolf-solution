# 8003.40 latest-LB optimization loop

- Leaderboard-verified baseline: `submission_base_8003.40.zip`
- Baseline SHA-256: `9bb795b4a2945882e98071350ee8333a914f27552ac8935570a20e2a57afd36f`
- Baseline score: `8003.40`
- Requested cumulative improvement: `+20.0`
- Target score: `8023.40`
- Root `submission.zip`, `best_score.json`, and score CSV files remain protected and unchanged.

## Baseline transition

The prior 8002.63 baseline was superseded by the LB-verified 8003.40 ZIP. The
new baseline changes tasks 073, 111, 122, 260, 271, 285, 289, and 359. None
overlaps the nine previously accepted rewrites, so those rewrites were rebased
without replacing any new-LB member.

## Wave 1 (superseded by safe Wave 3)

- Candidate: `submission_8003.40_wave1_policy95_meta.zip`
- SHA-256: `c829af1a2928fbbcdae3d13a38a8f38777e4c38c6a08c2c5ae51a3c4f0bd2c49`
- Accepted tasks: 013, 105, 132, 158, 344, 349, 358, 379, 398
- Total cost reduction: `54`
- Projected gain: `+0.13559602885163713`
- Projected score: `8003.535596028852`
- Remaining to target: `19.864403971148363`
- Team-validator comparison against the exact 8003.40 ZIP reports all nine tasks `ACCEPT_STRICT`.
- task344 uses the authorized policy95 rule: known 266/266, independent fresh dual-ORT 4972/5000 (99.44%), generator errors 0, and external execution errors 0.
- The other rewrites are generator-domain exact or incumbent raw-equivalent. They add no asymmetric runtime error. task158 and task379 retain only failures already present symmetrically in the incumbent on some arbitrary off-domain grids.
- Archive audit: 400 unique members, no missing/duplicate/oversize tasks, ZIP integrity clean, and original order/metadata/comment preserved.

## Wave 2 (rejected after Conv-bias UB audit)

- Rejected candidate: `submission_8003.40_wave2_policy95_meta.zip`
- SHA-256: `e77821cfd497d5583da4d79afc49bd872e918e6d0c3bcb4a319e6635a2e23984`
- Contained changes: Wave 1の9件 + task153 `237 -> 236`
- Total cost reduction: `55`
- Projected gain: `+0.1398243649611608`
- Projected score: `8003.539824364961`
- Remaining to target: `19.86017563503884`
- **Do not submit or extend this ZIP.** `scripts/golf/check_conv_bias.py` found task153 has a QLinearConv bias initializer of length 9 for 10 output channels. This is a known undefined/non-deterministic failure condition even though the sampled known/fresh checks passed.
- task153 is therefore rejected and its projected gain is withdrawn. The current safe aggregate remains Wave 1 (`8003.535596028852`, +0.13559602885163713).
- The baseline ZIP and Wave 1 both report zero Conv/QLinearConv/ConvTranspose bias-length issues.

## Wave 3 (current safe aggregate)

- Candidate: `submission_8003.40_wave3_safe_meta.zip`
- SHA-256: `338b6c968bb345780a570ec849f17b2fc0c1233c5bd0a000c67a035aeafb0cd7`
- Accepted tasks: Wave 1の9件 + task109 `406 -> 405`
- Total cost reduction: `55`
- Projected gain: `+0.13806212134683094`
- Projected score: `8003.538062121347`
- Remaining to target: `19.86193787865317`
- task109 is a computational-payload-identical annotation correction: known 266/266, independent dual-ORT fresh 5000/5000 with errors 0, external executable cases raw-identical with no asymmetric failure, strict shape/data propagation PASS, and Conv-family bias UB 0.
- The metadata-preserving builder reports only task109 changed relative to Wave 1. Member order, archive comment, all member metadata, and the other 399 payloads are unchanged. ZIP integrity is clean, all 400 tasks are present, and the full archive has zero Conv-family bias issues.
- `wave3_delta_compare.json` independently reports task109 `ACCEPT_STRICT` with projected gain `+0.0024660924951938057` over Wave 1.
- `wave3_compare.json` rechecks all ten changed tasks against the exact 8003.40 LB ZIP and reports 10/10 `ACCEPT_STRICT`, total cost reduction 55, and total projected gain `+0.13806212134683093`.
- An independent reviewer reran all Wave 3 checks with new seeds and reports `APPROVE` with 23/23 gates passing. It additionally confirmed the other 399 members have identical uncompressed and compressed bytes.

## Parallel resume results

- Archive rescreen: task109 accepted; task20/228 had no truthful cost reduction; task254 was rejected for a 33-input giant Einsum and 412/500 external mismatches.
- Exact scan: 400/400 tasks rescanned, zero safe adoptees. task048 failed the authorized 95% fresh gate (1818/2000 = 90.9%); task233/333 were rejected as private-risk dust/giant-floating-contraction changes.
- Changed-LB tasks: 34 candidates audited, zero adoptees. task073/260/271/359 variants failed known correctness or structural support; task111/122 candidates raised runtime errors; task285/289 shape-cloak paths were not retried.
- task359 true-rule rebuild: generator-exact reference passed known 266/266 and fresh 5000/5000, but a sound implementation has a 600–1200 byte histogram floor versus incumbent cost 24, so no candidate was emitted.
- SOUND local rebuilds (168/192/343/344): all references passed known and fresh 5000/5000, but sound controls cost 20403/18973/178/910 versus incumbents 416/1609/173/197 (Wave 1 task344=191); no cheaper sound replacement exists in this wave.
- High-cost range 150–400: 251 tasks ranked; task156/237/345 selected and references passed fresh 5000/5000. Nine serious attempts found no correct lower-cost candidate.

## Current search policy

- Recompute official-like runtime cost before trusting archived `static` filenames.
- Require known correctness, independent fresh accuracy >=95%, candidate runtime errors 0 on generated cases, strict shape/data propagation, and no asymmetric external failure.
- Require `scripts/golf/check_conv_bias.py` to report zero issues on every aggregate ZIP; reject any candidate whose Conv-family bias length is smaller than its output-channel count.
- Reject giant/multi-input floating-point `Einsum` contractions even when generator fresh tests pass; changing their operand count/order can be platform-sensitive. In this wave that excludes task254 (33 inputs) and task333 (36 -> 35 inputs).
- For catalogued private-risk tasks, do not accept dust gains that change floating contraction structure (task048/task233 remain unchanged unless a stronger proof and isolated LB evidence become available).
- Reject lookup/private-zero lineage, shape cloaking, undefined behavior, and any candidate whose truthful cost does not decrease.
