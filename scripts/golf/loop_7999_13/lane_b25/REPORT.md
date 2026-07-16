# B25 strict lane report — task232 / task369

## Outcome

- Verified score gain: **+0.00**
- Promotable winners: **0**
- Baseline: `submission_7999.13_wave16_candidate_meta.zip`
- Baseline SHA-256: `4014cbafea4862f67ebf5ff24be13149b45b333c95bfa680be7216f001a6bb3a`
- No aggregate, root submission, score CSV, or shared artifact was edited.

## True-rule evidence

| Task | Readable rule | Known | Fresh generator |
|---|---|---:|---:|
| 232 | In each occupied row, alternate the source color and gray 5 from the sole source cell to the right. | 266/266 | 5000/5000 |
| 369 | Recolor each orthogonally connected black component by size: 1→3, 2→2, 3→1. | 265/265 | 5000/5000 |

Both readable rules have zero known mismatches, zero fresh mismatches, and zero generation errors.

## task232

The incumbent is a single 11-input Einsum with cost 116 (memory 0, parameters 116). It passes all 266 known examples in both ORT disabled/default modes and has truthful static/runtime shapes.

- Generated and screened 134 strictly smaller latent-axis pruning candidates (costs 41–102).
- None passed the complete known gate in both ORT modes.
- A rank-3 model was also trained against all 384 generator-exhaustive atomic row cases. Its best state still had 1,761 wrong active cells and was rejected before fresh validation.
- Complete loose/history scan covered 564 files and 17 distinct hashes. There is no valid model below cost 116; five distinct historical models only tie at 116.

## task369

The incumbent reports cost 130 (memory 128, parameters 2), but this score is obtained through inherited false shape declarations:

- 69 declared/runtime shape mismatches were traced.
- Truthful one-example intermediate storage is 244,820 bytes, not 128 bytes.
- Disabled ORT passes all 265 known examples.
- Default ORT cannot create a session: `CenterCropPad` receives a one-element shape for two axes.

Every loose tie-cost family inspected inherits the shape-cloak pattern. The complete history scan covered 572 files and 24 distinct hashes and found no model below cost 130. A truthful local-rule rebuild necessarily exposes nontrivial 30×30 intermediates, so no strictly cheaper, error-free candidate was available. Nothing from this family was promoted.

## Strict decision

No candidate satisfies all of: real lower cost, true generator rule, full checker/strict shape inference, truthful runtime shapes, no new/enlarged giant Einsum, complete known correctness in both ORT modes with zero errors, and fresh-5000 correctness. Therefore `winner_manifest.json` is intentionally empty.

Primary evidence:

- `audit.json`
- `history_inventory.json`
- `candidate_screen.json`
- `rank3_training.json`
- `winner_manifest.json`
