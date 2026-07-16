# Cost 11–100: cost≤10 structural-pattern transplant

## Result

The immutable authority is `submission_base_8011.05.zip` (LB 8011.05,
SHA-256 `ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56`).
All 148 authority tasks at cost 11 through 100 were processed.

- Guaranteed-safe half-cost candidates: **0**.
- Guaranteed-safe strict-lower candidates: **0**.
- `POLICY95` half-cost candidate: **task202, 48 -> 20**.
- `POLICY95` strict-lower but not half-cost candidate: **task070, 66 -> 52**.
- No root archive, score CSV, score ledger, or `others/` output was changed.

## New cost≤10 pattern scan

The 21 authority tasks at cost at most 10 collapse to 17 unique ONNX graphs.
Thirteen finite/static graphs were admitted as templates: channel `Gather`,
output-only `Transpose`, output-only/input-only `Einsum`, finite
`ConvTranspose`, `RoiAlign`, and `MaxRoiPool`. Four unique graphs containing
nonfinite initializers were rejected before task evaluation.

The scan transplanted every admitted literal graph across the 148-task scope
and added:

- the complete finite generic output-only family used by the previous
  score-25 scan;
- per-task exact channel-map synthesis as a 10-parameter `Gather`;
- one-node `Einsum` initializer-factor subset ablations;
- finite contiguous `ConvTranspose` kernel crops with exact pad compensation;
- optional finite `ConvTranspose` bias ablations.

This produced **22,280 candidate-task evaluations**. No formula was exact even
on the full 12-case quick gate, so no candidate reached profiling/admission.
Machine evidence is in `pattern_scan.json`; the search is reproducible with
`scan_patterns.py`.

## New current-authority history scan

`scan_current_history.py` rescanned 9,155 loose ONNX paths against the current
8011.05 authority. It deduplicated 426 target/SHA pairs, repriced 197
theoretical strict-lower models, and found 15 known-exact lower rows. These 15
rows are duplicate lineages for only four tasks: 070, 202, 322, and 372.

- task322 20 -> 19 and task372 13 -> 12 are rejected. Both candidates contain
  nonfinite `ConvTranspose` kernels and a 9-element bias while the dynamic
  weight exposes 10 output channels. They rely on undefined short-bias reads.
- task070 cost50 is rejected because fresh cases contain positive raw values in
  `(0, 0.25)`; cost52 is the stable retained variant.
- task202 cost20 and task070 cost52 are the only structurally admissible lower
  lineages. Both are known private-zero families, so they stay in `POLICY95`,
  never the guaranteed-safe bucket.

Machine evidence is in `current_history_scan.json`.

## POLICY95 candidates

| task | authority -> candidate | half | fresh seed 1 | fresh seed 2 | score gain | class |
|---:|---:|:---:|---:|---:|---:|---|
| 202 | 48 -> 20 | yes | 97.40% | 96.65% | +0.875469 | private-zero lineage, 14-input non-giant Einsum |
| 070 | 66 -> 52 | no | 99.00% | 98.45% | +0.238411 | private-zero risk |

Each fresh stream has 2,000 generated cases. Prior independent evidence covers
four ORT settings (`ORT_DISABLE_ALL`/default x threads 1/4); all settings agree.
Both candidates have zero runtime errors, nonfinite outputs, shape mismatches,
or small positive values. The current lane reprofiled the exact candidate
bytes against 8011.05 and reconfirmed costs 20/52, complete known correctness,
finite initializers, static canonical I/O, and stable margins.

The combined conditional gain is **+1.1138797608**, or **8012.1638797608** if
both tasks receive leaderboard credit. Accuracy above 95% does not guarantee
private-LB credit, so these files are not safe replacements for the champion.

Files:

- `candidates/task202_policy95_cost20.onnx`
- `candidates/task070_policy95_cost52.onnx`
- `task202_policy95_audit_reused.json`
- `task070_policy95_audit_reused.json`
- `MANIFEST.json`

Reproduction:

```bash
.venv/bin/python scripts/golf/agent_cost11_100_lowcost_patterns_401/scan_patterns.py
.venv/bin/python scripts/golf/agent_cost11_100_lowcost_patterns_401/scan_current_history.py
```
