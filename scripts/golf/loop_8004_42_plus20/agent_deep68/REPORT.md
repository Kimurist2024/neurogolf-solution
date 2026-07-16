# deep68 — task285/366/286/233 SOUND rebuild and exact-rewrite audit

## Outcome

**Safe winners: 0.** No model is eligible for integration into the immutable
`submission_base_8005.16.zip` lineage. The protected submission and integration
ZIP were not modified.

The immutable member costs were measured with the same runtime-aware scorer:

| task | immutable cost | closest generator-sound / truthful result | result |
|---|---:|---:|---|
| 285 | 8623 | 14685 | reject: 6062 above threshold |
| 366 | 7987 | 13915 exact rebuild; 9465 truthful historical approximation | reject: both above threshold; approximation is not exact |
| 286 | 7481 | 54552 bounded full-row rebuild | reject: 47071 above threshold |
| 233 | 7308 | 105047 fully truthful control; compact exact-default models retain false shape metadata | reject |

Machine-readable final gates are in `finalist_audit.json` and
`winner_manifest.json`.

## True-rule analysis

The transforms were compiled from `inputs/sakana-gcg-2025/raw/taskNNN.py` and
the corresponding generator implementations documented by the existing
specification solvers.

- **task285 — bounded component/reflection.** Detect each 2x2 color hub, recover
  the connected shown creature quadrant, reflect its support into the four
  quadrants, and recolor each copy with the corresponding hub color. The
  generator bounds the creature to a 5x5 support, but exact hub detection,
  component extraction, color recovery, and rendering still have a measured
  truthful floor above the current shortcut model.
- **task366 — dynamic template matching.** Split the input into equal panels
  along the long axis, infer panel roles/background/fill colors, recover marked
  rectangles from the dense panel, match their sparse marker patterns in the
  other panel, and paint the matching rectangles while preserving marker cells.
  This is global matching with data-dependent colors, dimensions, and anchors.
- **task286 — global topology/flood fill.** Color 8 forms walls. Starting from
  the non-black/non-wall seed pair, recover the complete connected non-wall
  component and paint it with the two seed colors according to checkerboard
  parity. Exact connectivity across a 25x25 generator domain is the dominant
  cost.
- **task233 — global template/exact-cover.** Find the red field and its bbox,
  decode isolated external 3x3 clues, identify closed in-field 3x3 frames that
  match exact clue codes, resolve candidates in the generator's population and
  overlap order, paint them, and crop to the field. Population-only and local
  greedy formulations have generator-derived counterexamples.

These are not low-cost Type-A local rewrites. Tasks 366/286/233 are global
topology/template problems; task285 is bounded but still needs exact component
and hub logic.

## New exact rewrite work

`audit_exact_rewrites.py` applies only input-independent equivalences:
TensorProto-identical initializer aliasing and proven unary identities. It does
not use examples, coordinates, colors, or lookup tables.

For the exact, runtime-shape-truthful task285 rebuild, five duplicate
initializer groups were merged. This saved 14 parameters:

```text
14699 = 13016 memory + 1683 params
14685 = 13016 memory + 1669 params
```

The candidate is
`candidates/task285_true_dedup.onnx`, SHA-256
`3e10bc4d23b8692c0c52893ef140e0df45c96ae3c28f13651b19f59325eb7837`.
Full checker and strict data-propagating inference pass, runtime-shape tracing
reports zero mismatches, and both ORT modes pass all 265 known cases. Because
the merged initializers are byte-identical after erasing their names, this is a
formal all-input equivalence and inherits the source rebuild's 1000/1000 ONNX
fresh and 3000/3000 reference evidence. It is nevertheless 6062 cost above the
immutable member, so the expensive fresh gate was not rerun and it is not a
winner.

The same audit found no parameter reduction in the exact task366 and task286
rebuilds. It removed one Identity from the truthful task366 historical graph,
but the charged cost remained 9465. A task233 exact diagnostic yielded a
three-parameter dedup (`17007 -> 17004`), still far above the threshold and
still inherited false shape annotations; it is not eligible.

## Low-cost history rescreen

All retained history candidates were remeasured and checked under both
ORT_DISABLE_ALL and default ORT.

- **task285:** no retained model is both strictly below 8623 and known-perfect
  in both modes. The nearby 8732/8767 graphs are already above the threshold;
  default ORT also fails on the lower-looking lineage.
- **task366:** apparent costs 7646/7839/7957 are known-perfect, but have 97–107
  declared/runtime shape contradictions. Repairing the lowest computation's
  metadata raises its truthful cost to 9465. Its historical fresh accuracy is
  4685/4757 = 98.4864%, not exact, and the repaired cost is 1478 above the
  immutable member.
- **task286:** cost 7122 is checker-clean, known-perfect in both modes, and
  runtime-shape truthful, but fails fresh: 4318/5000 = 86.36% in the dual-mode
  audit (the strict audit was 4294/5000 = 85.88%). This is below the user's 90%
  admission floor and far below the private-zero guarantee.
- **task233:** cost 4936 is checker-clean and known-perfect in both modes, but
  uses `TfIdfVectorizer`/`GatherND` lookup behavior and scores 0/100 on fresh
  generator cases. It is a definite private-zero model.

## Gate decision

No candidate simultaneously satisfies strict lower cost, known100 in both ORT
modes, the fresh threshold, truthful runtime shapes, and the structural safety
rules. Private-zero admission is also impossible for every low-cost lead: none
has decoded-rule, dual fresh100, runtime0, truthful-shape proof.

Therefore `result.json` and `winner_manifest.json` contain no winner, and no
submission ZIP was created or changed by this lane.
