# high59 audit report

## Decision

**SAFE WINNERS: 0** for `task151,213,122,094,220,260,342,331`.

No candidate reached the fresh-validation stage. Every strictly cheaper candidate
failed the complete known set under the first fail-closed runtime gate, so there
is no candidate that can satisfy the required dual-runtime known100 condition.
No submission ZIP or protected root file was changed.

## Immutable baseline

- Archive: `submission_base_8005.16.zip`
- SHA256: `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`
- Static official-like costs from the archive:

| task | memory | params | cost |
|---:|---:|---:|---:|
| 151 | 92 | 12 | 104 |
| 213 | 20 | 83 | 103 |
| 122 | 99 | 2 | 101 |
| 094 | 0 | 100 | 100 |
| 220 | 0 | 100 | 100 |
| 260 | 0 | 100 | 100 |
| 342 | 0 | 96 | 96 |
| 331 | 92 | 2 | 94 |

These already-fixed baselines use shape-cloaked graphs; they were treated as
immutable exactly as instructed. They were not proposed as new candidates.

## Search coverage

The retained archive inventory was audited first, then every locally retained
`*.onnx` artifact whose path identifies one of the eight tasks was scanned. SHA256
deduplication left 84 task/model pairs including the eight repeated baseline
models, or 76 distinct non-baseline candidates.

| task | distinct non-baseline artifacts | minimum valid candidate cost | strictly lower |
|---:|---:|---:|---:|
| 094 | 2 | 2848 | 0 |
| 122 | 4 | 100 | 1 |
| 151 | 5 | 106 | 0 |
| 213 | 13 | 110 | 0 |
| 220 | 0 | — | 0 |
| 260 | 31 | 42 | 20 |
| 331 | 3 | 1844 | 0 |
| 342 | 18 | 126 | 0 |

## Rejections

- `task122`: the sole strict-lower artifact costs 100 versus 101, but produces
  runtime errors on all 266 known cases under ORT `DISABLE_ALL`.
- `task260`: all 20 strict-lower initializer-prune artifacts cost 42–99 versus
  100. None is known-perfect. Most score 0/266; the best reaches only 23/266.
- All other tasks: no locally retained artifact is strictly cheaper than its
  immutable baseline.

Because the pre-fresh safe set is empty, independent 2-seed fresh tests, true-rule
private-zero proof, all-known runtime trace-cost aggregation, Conv UB, and final
ZIP integration are correctly not run.

## Evidence

- `history_lead_audit.json`: retained archive inventory audit.
- `all_candidate_audit.json`: exhaustive local SHA-deduplicated artifact audit.
- `scripts/golf/loop_8003_40/agent_changed_resume/known/task260_*.json`:
  earlier full-known evidence matched to the task260 candidates by SHA256.

