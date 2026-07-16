# 8002.63 latest-LB optimization loop

- Leaderboard-verified baseline: `submission_base_8002.63.zip`
- Baseline SHA-256: `a2da30657f3798e861f369ac896f36722ff658ed3e468c4d55db9a04eefbccfc`
- Baseline score: `8002.63`
- Requested cumulative improvement: `+20.0`
- Target score: `8022.63`
- Root `submission.zip`, `best_score.json`, and score CSV files remain protected and unchanged.

## Wave 1 (superseded by Wave 2)

- Candidate: `submission_8002.63_wave1_safe_meta.zip`
- SHA-256: `0d95072a95dda0ee32494fff4b10fe7ddf56a4fe477ab5e76b9faac3f15e4223`
- Accepted tasks: task013 `743 -> 731`, task105 `199 -> 194`, task158 `7627 -> 7615`, task349 `3964 -> 3960`, task358 `161 -> 155`
- Total cost reduction: `39`
- Projected gain: `+0.08229268673069612`
- Projected score: `8002.71229268673`
- Remaining to target: `19.917707313269304`
- Archive audit: 400 unique members, no missing/duplicate/oversize tasks, ZIP integrity clean, and original order/metadata/comment preserved.

## Evidence summary

- task013: raw-bitwise exact on known and fresh dual-ORT 5000/5000; external random 500/500; truthful shapes and full checker.
- task105: raw-bitwise exact to the authority graph; known complete; fresh 4970/5000 (99.4%) in both ORT modes with runtime errors 0; external random 500/500. This uses the user-authorized >=95% exception without adding behavior.
- task158: exact permutation-mask reuse; known complete; fresh dual-ORT 5000/5000; all 33 reachable generator shapes observed; no asymmetric failure and no new giant Einsum.
- task349: global table identity over all 11 indices; known 267/267; fresh raw equality 20,000/20,000 across two seeds and both ORT modes; external arbitrary-random 500/500 raw equality; runtime errors 0.
- task358: known complete; fresh dual-ORT 5000/5000; runtime errors 0; external random 500/500; maximum Einsum inputs reduced from 44 to 42.

The candidate is byte-identical to the previously audited latest-base overlay,
but was rebuilt directly from the exact LB ZIP so the new campaign has an
independent baseline-pinned build audit.

## Wave 2 (superseded by Wave 3)

- Candidate: `submission_8002.63_wave2_policy95_meta.zip`
- SHA-256: `b37155edac856cdd9c3d4586cbccc1b2b6be80ef871c2abdb85ef4de73e794f4`
- Accepted tasks: Wave 1の5件 + task132 `316 -> 312`
- Total cost reduction: `43`
- Projected gain: `+0.0950317125081277`
- Projected score: `8002.725031712508`
- Remaining to target: `19.904968287491873`
- task132 is a one-node exact gauge/initializer-reuse rewrite. Known data is 267/267 in both ORT modes, two independent fresh runs total 10,000/10,000 per mode with runtime errors 0, full checker/strict truthful shapes pass, and the latest external arbitrary-random run is 500/500 with `ACCEPT_STRICT`.
- A prior arbitrary-random seed observed 499/500 threshold agreement. This remains above the user-authorized 95% policy and is not a runtime error; the generator-domain accuracy is 100% across both independent runs.
- Archive audit: 400 unique members, no missing/duplicate/oversize tasks, ZIP integrity clean, and original order/metadata/comment preserved.

## Wave 3 (superseded by Wave 4)

- Candidate: `submission_8002.63_wave3_policy95_meta.zip`
- SHA-256: `552b71eb693ca368b9e443855326d92c11fca1360c0c523b7900a07d51a13b54`
- Accepted tasks: Wave 2の6件 + task398 `350 -> 347`
- Total cost reduction: `46`
- Projected gain: `+0.10364008704472738`
- Projected score: `8002.733640087045`
- Remaining to target: `19.896359912955273`
- task398 keeps the incumbent 69-input Einsum unchanged and only reuses an existing carrier. Known data and two independent fresh runs are 100% in both ORT modes with runtime errors 0; full checker/strict truthful shapes pass; the latest external random run is 500/500 with `ACCEPT_STRICT`.
- A prior arbitrary-random seed observed 496/500 threshold agreement (99.2%). This satisfies the user-authorized 95% policy, is not a runtime error, and the task has no catalogued private-zero lineage.
- Archive audit: 400 unique members, no missing/duplicate/oversize tasks, ZIP integrity clean, and original order/metadata/comment preserved.

## Wave 4 (current audited aggregate)

- Candidate: `submission_8002.63_wave4_policy95_meta.zip`
- SHA-256: `d89773db19e0dab921a87b491e19a8e94d372852b7535b61267fba3bc488ce89`
- Accepted tasks: Wave 3の7件 + task344 `197 -> 191` + task379 `1951 -> 1949`
- Total cost reduction: `54`
- Projected gain: `+0.13559602885163713`
- Projected score: `8002.765596028852`
- Remaining to target: `19.864403971148363`
- task344 is the user-authorized local-rule policy95 candidate. Known data is 266/266 with errors 0, independent fresh dual-ORT is 4972/5000 (99.44%) in both modes, and generator errors are 0. External arbitrary-grid execution is 500/500 with no runtime errors; its behavior is intentionally not incumbent-equivalent outside the generator domain.
- task379 exactly factors a duplicated initializer mode. Known raw equality is 266/266, fresh raw equality is 5000/5000 in both ORT modes with candidate runtime errors 0, and fresh truth accuracy is 4999/5000 (99.98%). External random testing is raw-identical on all 499 executable cases; one case fails symmetrically in both incumbent and candidate, so the rewrite adds no error path.
- Wave 4 comparison reports both tasks `ACCEPT_STRICT`, with incremental projected gain `+0.03195594180690975`.
- Archive audit: 400 unique members, no missing/duplicate/oversize tasks, ZIP integrity clean, and original order/metadata/comment preserved.

## Current search lanes

- task080: non-quarantine exact/current-equivalent reduction search.
- task125: archived low-cost family plus truthful runtime-cost revalidation.
- task201: archived cost543/682 family plus exact reduction search.

No candidate is accepted if it introduces runtime errors, private-zero or
lookup lineage, shape cloaking, undefined behavior, or a known leaderboard
regression.
