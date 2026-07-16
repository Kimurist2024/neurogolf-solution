# new_mid31 audit report

## Outcome

No candidate is admissible. The eight exact `submission_base_8005.16.zip`
members were audited without modifying or rebuilding any submission archive.
Projected gain is `+0.0`.

| task | exact base cost (memory+params) | known dual ORT | decisive result |
|---:|---:|---:|---|
| 046 | 631 (401+230) | 267/267, runtime0 in both | no lower candidate |
| 157 | 853 (750+103) | 265/265, runtime0 in both | private-risk Type C; no sound lower candidate |
| 161 | 190 (120+70) | 266/266, runtime0 in both | historical 176/188 variants fail; fresh rule ambiguity |
| 189 | 183 (149+34) | disable-all 266/266; default 266 runtime errors | protected incumbent has shape cloak; no truthful dual candidate |
| 384 | 180 (162+18) | 266/266, runtime0 in both | protected incumbent has two shape cloaks; no truthful lower candidate |
| 193 | 170 (0+170) | 266/266, runtime0 in both | exhaustive smaller single-Conv search proved infeasible |
| 195 | 150 (129+21) | 265/265, runtime0 in both | protected incumbent has full-grid shape cloaks; no truthful lower candidate |
| 281 | 161 (42+119) | 266/266, runtime0 in both | giant 38-input Einsum plus shape cloaks; no clean lower candidate |

All models pass the ONNX full checker and strict shape inference with
`data_prop=True`; all use standard ONNX domains and have Conv-family bias UB0.
The runtime trace separately exposes shape cloaks which strict static inference
alone does not catch. These fixed baseline members were not re-offered as new
candidates.

## True rules

- task046: split the three-row path at blank columns; gray-5 endpoints encode
  inter-segment vertical offsets. Remove separator columns, align the segments,
  restore their segment colors, and concatenate them.
- task157: match each gray creature near the bottom to the partially hidden
  black footprint under the top red field, recolor that hidden creature blue,
  and erase the gray references. This is global template matching (Type C).
- task161: identify the laser color from its four border endpoints and draw its
  complete horizontal/vertical lines, discarding distractors.
- task189: orient the corner 2x2 palette and 6x6 green mask around the cyan
  divider, then recolor mask cells by palette quadrant.
- task384: tight-crop the yellow object and enlarge each source pixel to 2x2.
- task193: retain a colored cell only if it has same-color support on both a
  vertical and a horizontal neighbor; isolated static disappears.
- task195: tight-crop/downsample the enlarged 3x3 Conway sprite and emit its
  9x9 self-product (Kronecker pattern) in gray.
- task281: the cyan dot chooses a cardinal direction; extend the framed box to
  that dot, continuing the outer border and inner fill. Rotation makes the rule
  direction-independent.

The Sakana functions for tasks 046/161/189/384/193/195/281 match every stored
pair. On two independent 200-case fresh streams, all except task161 are
200/200 on both; task161 is 198/200 on both because the compressed selector can
collide with distractor colors. The task157 Sakana function is combinatorially
expensive and was stopped rather than treating an unfinished run as evidence;
its private-zero catalog status therefore requires a new sound implementation
and fresh100 proof before any future adoption.

## task193 lower-bound search

The only credible clean opportunity was the honest one-node task193 group
Conv. I collected all 266 stored cases and tested every kernel/padding
configuration with kernel height and width in 1..4 (100 configurations):

- Every kernel with area below 16, even with a learned per-channel bias,
  encounters an identical input patch that requires opposite output labels.
  Such a local Conv is impossible regardless of weights.
- For a 4x4 Conv without bias (cost 160), 15 padding placements have the same
  patch conflict. The incumbent placement `[2,2,1,1]` has no direct conflict,
  but its complete signed constraints are not linearly separable through the
  origin. Therefore removing the ten-element bias is impossible.

No ONNX candidate was emitted, so fresh candidate gating was correctly skipped.

## Evidence

- `baseline_audit.json`: SHA, exact structure, actual/declared cost evidence,
  dual-known results, checker/data-prop, UB, and runtime-shape traces.
- `true_rule_audit.json`: known and independent fresh comparisons for the seven
  tractable decoded functions.
- `task193_conv_search.json`: all 100 kernel/padding feasibility outcomes.
- `result.json`: machine-readable final disposition.

Protected files (`submission.zip`, `submission_base_8005.16.zip`,
`best_score.json`, `all_scores.csv`, and `a.csv`) were not changed. No ZIP was
created or integrated.
