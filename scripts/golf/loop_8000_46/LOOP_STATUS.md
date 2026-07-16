# 8000.46 strict optimization loop

- Exact leaderboard baseline: `submission_base_8000.46.zip`
- Baseline SHA-256: `74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534`
- Baseline archive: 400 unique tasks, ZIP integrity clean
- Byte-identical prior reference: `loop_7999_13/submission_7999.13_wave17_candidate_meta.zip`
- Requested improvement: `+20.0`
- Target score: `8020.46`
- Root `submission.zip`: protected and unchanged

## Wave 1 (safe aggregate)

- Candidate: `submission_8000.46_wave1_safe_meta.zip`
- SHA-256: `4cc03921e41b62dfc9f8c48fb4f4846d9ae8382275af167f9cb62f2f37da5600`
- Replacements: task013 `743 -> 739`, task105 `199 -> 195`
- Predicted gain: `+0.02570338993030319`
- Predicted score: `8000.48570338993`
- Remaining to target: `19.974296610069697`
- Build audit: order, metadata, comment, and 400-member completeness preserved
- External archive audit: valid, no missing/duplicate/oversize members
- Per-task evidence: task013 and task105 are raw-bitwise equal to the exact baseline on 5000 fresh cases under both disabled/default ORT, with zero runtime errors. Both pass known data, full checker, strict truthful shapes, and external task validation. The aggregate ZIP comparison also returns `ACCEPT_STRICT` for both tasks on 500/500 random cases.
- Rejected from the preliminary archive: task132 `316 -> 312` passed its earlier fixed-seed 5000/5000 audits but the new baseline-pinned ZIP comparison found one threshold mismatch at random case 350 (499/500). It is excluded because the rewrite is not exact-base-equivalent.
- This wave is not promoted to the protected root submission.

## Latest verified LB base overlay

The workspace also contains the later leaderboard-verified
`submission_base_8002.63.zip` (SHA-256
`a2da30657f3798e861f369ac896f36722ff658ed3e468c4d55db9a04eefbccfc`).
Its task013/105/158/358 members are byte-identical to the 8000.46 authority,
so the same four verified replacements were overlaid without conflict:

- Candidate: `submission_8002.63_wave2_safe_meta.zip`
- SHA-256: `0d95072a95dda0ee32494fff4b10fe7ddf56a4fe477ab5e76b9faac3f15e4223`
- Projected score: `8002.71229268673`
- Accepted overlay tasks: task013, task105, task158, task349, task358
- task349 relation-only rewrite: `3964 -> 3960`; all 11 table indices remain executable, known 267/267, fresh raw equality 20,000/20,000 across two seeds and both ORT modes, runtime errors 0, and external arbitrary-random 500/500 raw equality with `ACCEPT_STRICT`.
- Audit: 400 unique tasks, no missing/duplicate/oversize member, metadata/order/comment preserved, both Conv-bias scanners clean.
- The root `submission.zip`, score ledger, and both authority archives remain unchanged.

## Wave 2 (current audited aggregate)

- Candidate: `submission_8000.46_wave2_safe_meta.zip`
- SHA-256: `fa3b4b43ac7ccb4e3aad66e48192a66939a3ca601ddf18e4c68ae3bcb58bd08d`
- Accepted tasks: task013 `743 -> 739`, task105 `199 -> 195`, task358 `161 -> 155`
- Predicted cumulative gain: `+0.06368263799551965`
- Predicted score: `8000.523682637996`
- Remaining to target: `19.93631736200448`
- task358 evidence: known 265/265 and fresh 5000/5000 in both ORT modes, runtime errors 0, external validator `ACCEPT_STRICT`, truthful shapes, and maximum Einsum inputs reduced from 44 to 42.
- Archive audit: 400 unique tasks, no missing/duplicate/oversize members, order/metadata/comment preserved.
- This wave is not promoted to the protected root submission.

## Adoption gates

Every winner must be cheaper than this exact 8000.46 baseline (or the current audited aggregate), pass complete known data, meet the user-authorized >=95% fresh threshold with zero candidate runtime errors, and normally pass fresh/domain-generator cases at 100%. Exact raw-equivalent rewrites may use the >=95% exception only when they introduce no new behavior. Full checker, truthful shapes, both ORT modes, structural/UB audit, and external validation remain mandatory. No lookup memorization, shape cloaking, or new/enlarged unsafe giant Einsum is accepted.

## Wave 3

- Candidate: `submission_8000.46_wave3_safe_meta.zip`
- SHA-256: `1564387356057460d0851a83ed6eb91949a7661ed464de6bdf10b6f2745dd296`
- Accepted tasks: task013 `743 -> 739`, task105 `199 -> 195`, task358 `161 -> 155`, task398 `350 -> 347`
- Predicted cumulative gain: `+0.0722910125321194`
- Predicted score: `8000.532291012532`
- Remaining to target: `19.92770898746788`
- task398 evidence: known 268/268 and fresh 5000/5000 in both ORT modes, runtime errors 0, stable margin, truthful shapes, and external validator `ACCEPT_STRICT`.
- Archive audit: 400 unique tasks, no missing/duplicate/oversize members, order/metadata/comment preserved.
- This wave is not promoted to the protected root submission.

## Wave 4 (superseded after independent-seed rejection)

- Candidate: `submission_8000.46_wave4_safe_meta.zip`
- SHA-256: `3aa87e8f5e02222d1a5785362616600c9404891f83d2418a0349a499e1ab66ac`
- Accepted tasks: task013 `743 -> 739`, task105 `199 -> 195`, task158 `7627 -> 7615`, task358 `161 -> 155`, task398 `350 -> 347`
- Predicted cumulative gain: `+0.07386560936709703`
- Predicted score: `8000.533865609367`
- Remaining to target: `19.926134390632903`
- task158 evidence: exact permutation-mask reuse; known 266/266 and fresh 5000/5000 in both ORT modes, raw equal to the exact baseline, runtime errors 0, all 33 generator-reachable shapes observed, truthful runtime shapes, Conv UB 0, and external validator `ACCEPT_STRICT`.
- Archive audit: 400 unique tasks, no missing/duplicate/oversize members, ZIP integrity clean, order/metadata/comment preserved, and full-archive Conv bias UB count 0.
- This wave is not promoted to the protected root submission.

The later aggregate comparison with seed `80004604` rejected task398: the
cost-347 rewrite differed from the exact baseline on 4/500 arbitrary random
cases. Wave 4 is therefore retained only as evidence and is not the current
safe candidate.

## Wave 5 (superseded by Wave 6)

- Candidate: `submission_8000.46_wave5_safe_meta.zip`
- SHA-256: `5a5e0c77d2942232bfb494e33f1e2f8fc42b943c359032fda89683e9c2f14452`
- Accepted tasks: task013 `743 -> 731`, task105 `199 -> 194`, task158 `7627 -> 7615`, task358 `161 -> 155`
- Total cost reduction: `35`
- Predicted cumulative gain: `+0.08128309552934354`
- Predicted score: `8000.541283095529`
- Remaining to target: `19.918716904471`
- Aggregate comparison: all four tasks return `ACCEPT_STRICT` on 500 random cases at seed `80004605`; task158 has generator-domain errors 0 and only symmetric off-domain failures.
- task013: known raw-bitwise exact in both ORT modes, fresh 5000/5000, truthful shapes, external 500/500; its shared reduction lowers cost from the Wave-4 value 739 to 731.
- task105: raw-bitwise exact to the authority model, known complete, fresh 4970/5000 (99.4%) in both ORT modes with runtime errors 0, external 500/500; this meets the user-authorized >=95% gate and lowers cost from 195 to 194.
- Archive audit: 400 unique tasks, no missing/duplicate/oversize members, ZIP integrity clean, order/metadata/comment preserved, and both full-archive Conv bias scanners report 0 findings.
- task398 is deliberately absent; its independent-seed mismatch evidence is in `wave4_changed_tasks_compare.json`.
- This wave is not promoted to the protected root submission.

## Wave 6 (current audited aggregate)

- Candidate: `submission_8000.46_wave6_safe_meta.zip`
- SHA-256: `ff70bbe71b1d40ccf5db0105265b1c835dcdef4a26bed4fab002166220171414`
- Accepted tasks: task013 `743 -> 731`, task105 `199 -> 194`, task158 `7627 -> 7615`, task349 `3964 -> 3960`, task358 `161 -> 155`
- Total cost reduction: `39`
- Predicted cumulative gain: `+0.08229268673069612`
- Predicted score: `8000.542292686731`
- Remaining to target: `19.917707313269304`
- task349 uses a global integer identity over all 11 stored indices and does not shrink the executable domain. It passes full checker, strict shape inference, known 267/267, dual-ORT fresh raw equality 20,000/20,000, errors 0, and external random 500/500 raw equality with `ACCEPT_STRICT`.
- Archive audit: 400 unique tasks, no missing/duplicate/oversize members, order/metadata/comment preserved, ZIP integrity clean.
- The shorter-table task349 experiment was rejected because it caused off-domain Gather errors; it is not present in this ZIP.
- This wave is not promoted to the protected root submission.
