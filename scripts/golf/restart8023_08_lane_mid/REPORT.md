# 8023.08 mid-cost evidence lane

## Authority and admission gate

- Immutable authority: `submission_base_8023.08.zip`
- SHA-256: `0e29e8d57f7ac58136a9574351c9c6f3056f9debf6eeee9c181c8f2e9fac690a`
- Scope: current ledger cost 250 through 399.
- Admission requires official/local gold exact, full checker plus strict/static
  inference, stable raw margin, fresh 2,000 x 2 at 100% in four ORT
  configurations, and zero runtime errors/nonfinite outputs/output-shape
  mismatches/small-positive outputs.
- The lane never wrote `submission.zip`, `all_scores.csv`, or
  `best_score.json`.

## Unique result

**No new unique strict winner was found.**

The history worker independently rediscovered task200 cost 342 from the
concurrent high-cost lane.  It is deliberately not counted as a new mid-lane
result:

| task | authority | candidate | gain | status |
|---:|---:|---:|---:|---|
| 200 | 346 | 342 | +0.0116280380 | strict pass, duplicate of `restart8023_08_lane_high` |

Duplicate candidate:
`candidates/task200_POLICY90_cost342_c659ae401e4c.onnx`

SHA-256:
`c659ae401e4c92a53cad5d4a251aac3ad562c7e919dcd5b4b82c42ed63c8a07d`

Its independent evidence in `worker_2.json` is gold 84/84 exact, full and
strict/static pass, fresh 2,000/2,000 x2 in all four ORT configurations,
minimum positive 1, every error/stability counter zero, and a non-mutating
official verifier result of `correct=true`, cost 342.

## Exhaustion evidence

- Three history/exact/transfer workers covered 36 eligible tasks.  Workers 0
  and 1 had no finalist; worker 2 had only the duplicate task200 result.
- Exact serialized dedup/static folding considered 31 variants; all 31 failed
  preflight and none reached admission.
- Exact Einsum outer-factor/dedup scanned all 44 band tasks; seven models were
  structurally factorable but global cost optimization produced no finalist.
- Rank>=2 census covered 44 tasks, 92 Einsum nodes, 142 constant operands and
  226 axis partitions.  The four apparent parameter-saving partitions all
  belong to task398 and all fail the available-Einsum-label budget, leaving
  zero structural candidates.
- A second broad exact-rewrite pass covered the safe cost-250..299 subset in
  three partitions.  It produced five mechanical variants and zero
  known-exact finalists.
- Twenty-six semantics-preserving ONNX optimizer passes were also applied to
  each of 11 safe cost-250..299 tasks.  No valid model had lower cost.  Scorer
  `-1` rejection sentinels are treated as errors, never as low costs.  Old
  `*_cost-1.onnx` artifacts from the initial census are explicitly marked
  `DO_NOT_USE.md` and are not candidates.
- task345 cost 369 is present in the authority.  No history, exact, transfer,
  factor, or optimizer pass found a cheaper admissible successor.

## Root protection note

`submission_base_8023.08.zip` retained its required authority hash.  The three
mutable root files (`submission.zip`, `all_scores.csv`, and `best_score.json`)
were changed externally while this lane ran; none of those writes came from
these scripts.
