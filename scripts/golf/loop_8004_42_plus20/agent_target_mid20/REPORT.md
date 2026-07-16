# +20 target-mid20 audit

## Outcome

No model is safe to promote. The accepted-candidate set is empty, projected gain is `+0.000000`, and no ZIP/CSV/root artifact was modified.

The named authoritative baseline is `submission_base_8005.16.zip` (SHA-256 `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`). Task245 is a material memo mismatch: its member measures **387 = memory 304 + params 83**, not 291. All comparisons below use the archive member and official local profiling path.

## Current members

| Task | SHA-256 (prefix) | Cost | Disabled/default known | Declared → observed | Disqualifier / disposition |
|---:|---|---:|---|---|---|
| 051 | `67222f7b5521` | 279 | 265/265, 265/265 | full → full | TfIdf lookup; 64-input giant Einsum |
| 064 | `56a992e221f0` | 271 | 267/267, 267/267 | full → full | internal CenterCropPad cloak; 58-input giant Einsum |
| 185 | `b9cdf53f2708` | 279 | 267/267, 267/267 | full → full | 8 TfIdf lookups; private/history-black task |
| 200 | `8c91d1a61ac9` | 346 | 84/84, 84/84 | full → full | structurally safe, but no cheaper safe equivalent |
| 245 | `e22d0d661df9` | 387 | 267/267, 267/267 | `[1,1,1,1]` → full | 7 CenterCropPad nodes; false output shape |
| 264 | `d2be90cfe369` | 362 | 265/265, session failure | `[1,10,1,1]` → full | 43 CenterCropPad nodes; false shape; default ORT failure |
| 394 | `cb47909c49db` | 350 | 266/266, 266/266 | `[1,1,30,30]` → full | CenterCropPad output-shape cloak |
| 397 | `55ff7a949d67` | 364 | 266/266, session failure | `[1,1,1,1]` → full | 13 CenterCropPad nodes, TfIdf lookup, false shape, default ORT failure |

Task200's incumbent is the only clean member of this set: standard domain, truthful `[1,10,30,30]`, strict shape inference with `data_prop=True`, no lookup or CenterCropPad, maximum five Einsum inputs, and exact Conv bias length 2 for two output channels.

## True-rule audit

The independent solvers in `verify_true_rules.py` match each generator's `validate()` cases and both fresh seeds (`80051620`, `80061620`) at 5000/5000 each:

| Task | Generator | Decoded rule |
|---:|---|---|
| 051 | `25d487eb` | Extend the rare beam color through the triangular laser's narrow side to the boundary, rotation-normalized. |
| 064 | `2c608aff` | Project aligned sparse dots toward the solid rectangle; preserve diagonal dots. |
| 185 | `7837ac64` | Recover the colored 3x3 pattern from intersections of the line grid. |
| 200 | `8403a5d5` | From the bottom marker, draw alternating full-height color stripes and gray turn endpoints. |
| 245 | `a1570a43` | Translate the red 5x5 sprite into the green-corner 7x7 frame, including diagonal displacement. |
| 264 | `a8c38be5` | Decode nine gray 3x3 digit glyphs and place their colored strokes into digit slots of a 9x9 grid. |
| 394 | `f9012d9b` | Recover the black square bite from the periodic pattern using the same-residue row one period away. |
| 397 | `fcc82909` | Add a green shadow to each 2x2 box with height equal to its number of distinct colors. |

Task264 uses seeded constructive layouts passed into the real generator because its default rejection sampler is pathologically slow. The construction enforces the generator's same size, color, and one-cell nonoverlap constraints; both 5000-case sets are generator-legal.

## Candidate and history decisions

Task200 received the only plausible local reductions:

- Historical 344 SHA `53802c…` is 0/84 known (and 0/5 fresh probe): permuting the latent basis is invalid because ScatterElements changes only row 0.
- New legal 344 SHA `b7b2a214…` has cost `200 + 144`, truthful shape, strict `data_prop`, standard domain, and no Conv bias UB, but is 0/84 under both ORT modes. It reproduces the visual argmax while leaving required background-channel one-hot values at zero; the official `> 0` mask therefore rejects it.
- Historical 345 SHAs `b9036595…` and `a54e5330…` use a length-1 bias for a two-channel Conv. Both are UB-dependent; `b9036595…` was also only 4999/5000 in the retained fresh run.

The broad history scan found no remaining prospect: task051/064 floors are 1744/8356; task245's best floor is 406 versus authoritative 387; task264's best non-giant floor is 461 and its nominal 358 candidate fails runtime; task394's floor-189 file measures 351 and old 342/348 files are known-wrong; task397's apparent floors 269–298 actually measure 368–409. Task185 was intentionally excluded by the private/history-black catalog, so only a new decoded-rule model could qualify.

Full machine-readable evidence is in `result.json`, `current_audit.json`, `current_anatomy.json`, `rule_full/`, and `rejected/task200_zero_background_cost344.json`.
