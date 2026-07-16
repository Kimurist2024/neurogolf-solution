# Lane harvest report — baseline 7999.13

## Result

One candidate is safe to hand back: `task109` reduces exact member cost from 406 to
405, for a projected score gain of **+0.0024660925**. The projected aggregate score
is **7999.1324660925**. No root submission, score ledger, or artifact was modified.

Candidate:

- `scripts/golf/loop_7999_13/lane_harvest/winner_task109.onnx`
- source: `others/7907/task109_improved(1).onnx`
- SHA-256: `2e7be8671e2e8abe9d3f2f77f0b068f54a70a584ce477affb28fee6372bd25ef`

## Validation

The baseline authority was the exact contents of `submission_base_7999.13.zip`
(SHA-256 `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`),
not `all_scores.csv`.

`task109` passed:

- ONNX checker, strict static shape inference, canonical I/O/domain, and structural gates
- ORT 1.24 with graph optimizations disabled
- complete known validation: 266/266 correct, zero skipped/errors
- `check_conv_bias`: zero findings
- margin stability: pass, minimum margin 11.0
- independent team-validator differential: `ACCEPT_STRICT`

The differential requested 3000 generated inputs. Both baseline and candidate were
executable on 260 and matched exactly on all 260 in raw and thresholded output. Both
failed identically on the other 2740; there were no asymmetric failures. This limited
executable coverage is disclosed because the gain is small, although the independent
validator still returned `ACCEPT_STRICT` and all executable outputs were bit-identical.

The protobuf comparison gives an additional semantic guarantee: after clearing
`graph.value_info`, baseline and candidate serialize identically. The sole change is the
non-executable annotation for `state_rows_pad`, from shape `[1,1,1,2]` to
`[1,1,1,1]`; nodes, initializers, graph I/O, and opsets are unchanged. Therefore the
candidate cannot alter runtime computation even on generated inputs that both models
reject.

## Search coverage

Thirteen historical source roots were scanned, covering 2,162 loose-model observations
and 10,000 members from 27 ZIPs. After SHA deduplication, exact-baseline duplicate
removal, and task exclusions, 1,134 different candidates across 372 tasks were screened.

The pipeline applied static cost floors, checker/shape/structure checks,
`check_conv_bias`, ORT-disabled execution, and complete known-gold validation before
running 3000-case independent differential validation on finalists. Baseline scoring
timed out for tasks 50, 287, 315, 328, 358, and 359, so no candidate from those tasks
was accepted by this lane.

## Rejections and exclusions

Known-gold-only winners for tasks 23, 90, 118, 131, 191, 268, 365, and 366 were
rejected because fresh differential tests exposed threshold-output or execution
differences. Task219 was removed after harvest because it is a confirmed private-zero
task in `docs/golf/private_zero_tasks.md`. These models remain only as audit evidence;
`winner_manifest.json` accepts task109 alone.

Machine-readable details are in `winner_manifest.json`, `scan_results.json`, and the
`external_taskNNN.json` reports.
