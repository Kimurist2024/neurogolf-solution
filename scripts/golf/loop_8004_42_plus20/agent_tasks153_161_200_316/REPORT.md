# tasks153/161/200/316 SOUND exact-regolf

## Outcome

`winner: null`

No admissible strictly-lower candidate was found.  Projected gain is `+0.0`.
Nothing was copied to `others/71407`, and this lane did not modify the root
submission, score ledger, or any baseline archive.

The immutable authority is `submission_base_8009.46.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.

| task | authority SHA-256 | memory | params | cost | exact-cleanup result |
|---:|:---|---:|---:|---:|:---|
| 153 | `a0fae5a7d5d934d8e9cd382da50b6cdb8c1995a0cb991174631d17abe6247159` | 154 | 76 | 230 | no changed profile |
| 161 | `5dc274d8515f1ac2a5c58583197984cd60fa2ede69fbe8206992f98940a38fbe` | 120 | 70 | 190 | no changed profile |
| 200 | `8c91d1a61ac9bdd4cb5e812b5f0db57e0bf49c1cf2fd256724960118430df8f3` | 200 | 146 | 346 | no changed profile |
| 316 | `3c84610b001e252e3861203767b5a60682784a26c42f2fce111c73bb3b391640` | 131 | 115 | 246 | no changed profile |

## Generator rules

- **task153 / `681b3aeb`**: two separated colored 3x3 partial creatures are
  complementary halves of one 3x3 creature.  Normalize their translations,
  resolve which normalized bitmap belongs to which color, and overlay them in
  the 3x3 output.
- **task161 / `6cdd2623`**: the laser color is the color occurring as a paired
  endpoint on both sides of a row and/or both ends of a column.  Paint the two
  corresponding full laser lines.  The frequently used shortcut "choose the
  least frequent nonzero color" is not generator-total because a dust color
  can tie it.
- **task200 / `8403a5d5`**: from the bottom-row marker, paint alternating
  full-height stripes of its color while moving right; paint gray at each
  top/bottom turn endpoint.
- **task316 / `cdecee7f`**: collect the one nonzero color from each occupied
  column in left-to-right order, place colors 1..3 in the first output row,
  colors 4..6 reversed in the second, and colors 7..9 in the third.

The compact raw rules match every known case.  For tasks153/200/316 they also
match every fresh generated case.  For task161, the raw least-frequency rule
and the authority network expose the same generator-corner weakness described
below; the generator itself remains the output authority.

## Runtime and correctness audit

The first complete audit used all known cases and two seeds with 3000 fresh
instances per seed.  Every model was run in four configurations:
ORT_DISABLE_ALL/default x threads 1/4.  Raw tensors were compared bitwise to
ORT_DISABLE_ALL/threads1.

| task | known per config | fresh seed 153161200 | fresh seed 316200161 | cross-config raw | errors / nonfinite / bad margin |
|---:|---:|---:|---:|:---|:---|
| 153 | 265/265 | 3000/3000 | 3000/3000 | identical on every run | 0 / 0 / 0 |
| 161 | 266/266 | **2984/3000** | **2982/3000** | identical on every run | 0 / 0 / 0 |
| 200 | 84/84 | 3000/3000 | 3000/3000 | identical on every run | 0 / 0 / 0 |
| 316 | 266/266 | 3000/3000 | 3000/3000 | identical on every run | 0 / 0 / 0 |

Tasks153/200/316 have minimum positive margin `1.0`; task161 has minimum
positive margin above `0.973`.  Maximum nonpositive output is `0.0` for all
four models.  All outputs are `[1,10,30,30]`.

### task161 fail-closed finding

The authority member is not generator-exact despite its perfect known score.
It fails 34/6000 fresh examples identically in all four runtime
configurations.  A concrete legal failure at seed `153161200`, index 326 has
nonzero color counts `{2:4, 5:12, 7:4}`.  Color 2 is the true paired-endpoint
laser color, but the compact least-frequency selection chooses the earlier
tied color 7 and changes 106 output bits.  Concrete counterexamples are in
`evidence/task161_counterexamples.json`.

This does **not** authorize a behavior-changing replacement: the task brief
requires a strictly cheaper generator-exact candidate.  None exists in this
lane, so task161 is not staged.

## Structural and exact-cleanup gates

All four authority members pass full ONNX checker and strict shape inference
with data propagation.  Runtime exposure of every inferred intermediate found
zero declared/runtime contradictions:

- task153: 34 traced tensors;
- task161: 3 traced tensors;
- task200: 9 traced tensors;
- task316: 26 traced tensors.

All use standard domains, have no banned op, lookup op, `CenterCropPad`, giant
node, nested graph, dead node output, unused initializer, duplicate full
initializer, or Conv/QLinearConv bias-length UB.

Eleven fixed-point exact profiles were applied to every authority member (44
profiles): dead-end elimination, CSE, initializer alias/unused elimination,
idempotent and no-op elimination, Conv/Pad fusions, shape folds,
Einsum-to-MatMul, Where rewrite, and Add adjustment.  Every profile was
byte-unchanged; strict-lower count is zero.

The remaining tempting historical/local reductions are already decisive:

- task153's two shifted-square `Add(+254)` nodes cannot be removed while
  retaining its one-QLinearConv decoder.  Exhaustive int8 coefficient search
  finds no separating weight pair for colors 1, 2, and 8.  See
  `agent_8009_exact_A_115/audit/result.json`.
- task161's known cost-186 lead misses 1/266 known cases in every runtime
  configuration (`agent_mid20d_88/REPORT.md`).
- task200's legal cost-344 lead is 0/84 known because it loses required
  positive background-channel values
  (`agent_target_mid20/rejected/task200_zero_background_cost344.json`).
- task316's repository-wide screen found no actual candidate below 246; its
  cheapest policy-clean safe screen was cost 255
  (`scripts/golf/scratch_codex_plus10/wave3_c/REPORT.md`).

## Evidence

- `audit_lane.py` — structural, optimizer, known, fresh, four-runtime, raw,
  margin, and error audit.  The complete run results are summarized above; a
  requested early stop during a redundant re-run left `evidence/audit.json`
  as the retained partial machine log.
- `extract_task161_counterexamples.py` and
  `evidence/task161_counterexamples.json` — deterministic generator-legal
  task161 failures.
- `inspect_models.py` — initializer and node-level anatomy dump.
- `winner_manifest.json` — explicit null result.

