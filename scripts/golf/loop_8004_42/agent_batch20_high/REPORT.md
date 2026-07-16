# Batch20 high lane — task201–400

## Outcome

The requested ten additional high-range tasks were selected and frozen in
`TASKS.txt` before candidate exploration. No candidate satisfies every safety
and cost gate, so this lane contributes **0 accepted models** and projected
gain **+0.0**. No ZIP was built or merged, and none of the 27 fixed members was
changed.

Baseline authority:
`scripts/golf/loop_8004_42/submission_8004.42_fixed_rebase_meta.zip`.

## Frozen ten-task set

| Task | Baseline cost | Best lead | Decision |
|---:|---:|---|---|
| 374 | 481 | CastLike→Cast, actual cost 876 | Reject: costlier; no truthful model below 481 |
| 250 | 468 | ConvInteger, cost 467 | Reject: 0/265 known-correct |
| 324 | 439 | no lead below current quarter synthesis | Reject: no new safe lead |
| 308 | 434 | exact CastLike alias, cost 434 | Reject: tie; alternate TopK rewrite is runtime-invalid |
| 275 | 428 | no lower factor after shared-gate rewrite | Reject: no new safe lead; giant-float contraction changes forbidden |
| 338 | 426 | Boolean fusion / anchor removal | Reject: runtime error / actual cost 18423 |
| 333 | 423 | sign/gauge absorption, cost 421 | Reject: modifies 36-input floating Einsum |
| 268 | 422 | historical rebuild, cost 327 | Reject: fresh 2219/5000 (44.38%) and lookup construction |
| 377 | 409 | `diff5` witness reuse, cost 408 | Reject: default ORT 266/266 runtime errors and shape cloak |
| 279 | 397 | no further DCE/dedup rewrite | Reject: no new safe lead; current graph has 203 runtime-shape mismatches |

## Fresh current-baseline audit

Nine current members were re-profiled against the complete known corpus in
both ORT modes, with full checker, strict shape/data propagation, actual runtime
shape tracing, and Conv-family bias checks. The tenth, task333, is byte-identical
to the SHA already covered by the prior 265/265 plus raw-2000 exact audit; rerunning
its 36-input floating Einsum was stopped because the only candidate is forbidden
independently of accuracy.

Important findings that prevent unsafe reuse:

- task250 and task275 are truthful on the runtime-shape trace and run the full
  known corpus without errors in both modes, but neither has a new lower-cost
  admissible rewrite.
- task374 has 9 declared/actual shape mismatches. Its algebraic one-parameter
  rewrite exposes the true carrier and raises actual cost to 876.
- task324 and task308 have respectively 5 and 26 runtime-shape mismatches and
  fail default-ORT session construction at TopK.
- task338 and task377 cannot complete a truthful runtime trace; their buffer
  shapes conflict, and default ORT is also terminally invalid.
- task268 has 32 runtime-shape mismatches, `TfIdfVectorizer`, and
  `CenterCropPad`; its cheaper historical rebuild fails the mandatory fresh
  gate by a wide margin.
- task279 is complete-known correct in both modes, but tracing finds 203 false
  declarations. The global exact scan already removed its reachable dead nodes;
  carrying the cloak into a new candidate is not allowed.

All freshly audited current models pass ONNX full checking and strict symbolic
shape/data-propagation inference, and no unsafe Conv bias was found. Those checks
do not override actual runtime-shape failures, default-runtime errors, fresh
failure, lookup use, or the giant floating-Einsum prohibition.

## Evidence

- Machine-readable final dispositions: `RESULTS.json`
- Fresh current-member audit: `CURRENT_AUDIT.json`
- Frozen selection: `TASKS.txt`
- Reproducible audit driver: `audit_current.py`
- Prior identical-SHA exact task333 evidence:
  `../../loop_8003_40/agent_exact_resume/FINAL_REPORT.json`
- Historical strict candidate evidence:
  `../../loop_7999_13/lane_a28/REPORT.md`,
  `../../loop_7999_13/lane_a29/REPORT.md`,
  `../../loop_7999_13/lane_a5/REPORT.md`,
  `../../loop_7999_13/lane_b5/REPORT.md`,
  `../../loop_7999_13/lane_b21/REPORT.md`, and
  `../../loop_7999_13/lane_c27/REPORT.md`.

No candidate was copied to a winner directory because the accepted set is
empty. The protected root files and aggregate ZIP remain untouched.
