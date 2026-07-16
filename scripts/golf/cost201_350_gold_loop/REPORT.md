# Cost 201–350 gold-strict search (8012.23 authority)

## Result

- New admitted candidates: **0**
- Verified gain: **+0.000000**
- Authority score remains **8012.23**.
- `submission.zip` was not changed and remains byte-identical to
  `submission_base_8012.23.zip` (SHA-256
  `720ebf75d826945250e3c7d7ea11780a950d8d3038546e9c7595503277a1189f`).
- `all_scores.csv` was not changed (SHA-256
  `9a300740226178fe253c39191d193609193a0a2f69aded3487be274107231d19`).

Admission required all of the following: complete known train/test/arc-gen gold,
no runtime/shape/nonfinite/small-positive failure, structural audit, and 2,000
fresh generator cases at 100% (four ORT configurations).  The repository
`try_candidate.py` official-gold gate was reserved for candidates surviving
those earlier gates.  No model reached that final stage.

## Scope

The CSV cost band contained 43 tasks.  Known black/private-zero tasks and the
six newly confirmed failed-gold tasks (12, 110, 161, 175, 188, 355) were
excluded.  The remaining 37 tasks were divided across three disjoint workers.
Four members have legacy CSV/local-profile cost differences; the search used
the lower freshly reprofiled authority cost as the comparison floor, preventing
stale ledger values from creating false gains.

## Searches

1. History/archive rebase + current-graph simplification + low-cost templates
   (`worker_0.json`, `worker_1.json`, `worker_2.json`)
   - 21,223 candidate encounters
   - 13,775 matching ZIP members and 1,431 loose history models inspected
   - one cheaper known-gold survivor: task143 cost 212→148
   - task143 was rejected decisively on both fresh seeds: 1/2,000 correct
     (0.05%) in every ORT configuration
   - admitted finalists: 0

2. Internal tensor-rank ablation (`einsum_rank_report.json`)
   - task132, task199, task212
   - 31 one-component bond deletions
   - all 31 failed complete known gold

3. Same-dtype/same-static-shape node bypass (`same_shape_bypass_report.json`)
   - 483 encounters / 474 SHA-unique attempts
   - 343 structural-preflight rejects, 20 non-improvements, 111 known-gold
     rejects, 9 duplicates
   - complete-known survivors: 0

4. task075 constant-Identity bypass (`exact_builds.json`)
   - rejected before execution: deleting the Identity exposes a scalar shape
     initializer and makes the CenterCropPad graph fail strict shape checking

The `candidates/` directory is intentionally empty.  Rejected structural
artifacts are isolated under `rejected/` and must not be merged.
