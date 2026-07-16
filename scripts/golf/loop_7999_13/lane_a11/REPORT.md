# A11 audit report

- Exact baseline: `submission_base_7999.13.zip`
- Baseline SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Tasks: 065, 088, 105, 189, 224, 240
- Candidates screened: 28
- Safe winners: 0
- Verified score gain: `+0.000000`

## Task 105 priority audit

The 199-to-198 candidate is a real executable edit, not a metadata or
`value_info` deletion.  It deletes the one-element `one_f` initializer and
retargets six Einsum equations to the fixed batch label.  Both models have zero
`value_info`, and clearing metadata does not erase their difference.

The candidate passes the complete known corpus and profiles at memory 89 plus
109 parameters, cost 198.  It is nevertheless rejected.  The independent
fresh audit produced 4980/5000 in `ORT_DISABLE_ALL` and 4980/5000 in default
ORT, with 20 failures in each mode and zero generation errors.  It also retains
a 45-input giant Einsum.  Evidence is in `task105_summary.json`,
`task105_diff.json`, and the independent
`lane_archive_top200/task105_dual5000.json` result.

## Other retained/history candidates

| Task | Baseline | Candidates | Result |
|---:|---:|---:|---|
| 065 | 199 | 8 | Seven known-correct candidates cost 202 or more; one mismatched/errored. |
| 088 | 230 | 8 | The one cheaper profile (228) is incorrect; the only known-correct retained candidate costs 342. |
| 189 | 206 | 2 | Both are known-correct but cost 212 and 218. |
| 224 | 162 | 4 | All inherit a 62-input giant Einsum and fail the structure gate. |
| 240 | 172 | 5 | All use a 41-input giant Einsum and fail the structure gate. |

The complete per-candidate profiles and verdicts are in
`retained_scan.json`.  No model was promoted to fresh validation, and no root
ZIP, CSV, score pointer, or ledger was changed by this lane.
