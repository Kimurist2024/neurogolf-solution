# Wave B3 final report

The exact `7999.13` archive was used as the only cost baseline. No candidate in
this lane satisfies both the soundness gates and strict-cheaper requirement, so
the accepted winner list is empty and no root submission artifact was changed.

## Priority candidate: task208

The proposed `task208_single_vi1.onnx` nominally scores `1422 -> 1392`, but is
not executable under the target runtime:

- known fixtures: **0/266**, with **266 runtime errors**;
- direct generator fresh: **0/3000**, with **3000 runtime errors**;
- independent generator differential: **3000/3000 candidate-only failures**;
- failure: `Slice` allocator shape mismatch, `{1} != {16}`;
- checker, strict shape inference, static/banned/nested/function/sequence/sparse
  and bias checks pass, demonstrating that those checks alone do not detect the
  runtime allocator hazard;
- after clearing `graph.value_info`, the baseline and candidate executable
  graphs are byte-identical. The metadata change itself alters ORT buffer reuse.

The exact task208 baseline also fails the direct generator on case 6, and
`docs/golf/private_zero_tasks.md` explicitly records task208 as a private-zero
regression. The candidate is therefore rejected on three independent grounds.

## Generator rules and remaining tasks

- **task089**: complete the marker-bearing 3x3 sprite at repeated placements;
  red copies are horizontally mirrored and green copies are not. The exact
  baseline fails fresh case 272. Its `grn_crop3` VI shave is even worse: 0/267
  known, 3000/3000 candidate-only runtime failures from a QLinearConv allocator
  mismatch. Sound rule rebuilds are above the exact cost 1361.
- **task196**: recolor only complete blue rectangle borders of width and height
  at least 3; thin or gapped components stay blue. The exact cost-1210 compact
  graph fails fresh case 7. The known fully spec-derived packed-bitset rebuild
  is sound but costs 5573.
- **task255**: reveal hidden green interiors of artery/vein rectangles. This is
  information-theoretically ambiguous because legal edge-clipped generator
  states can have equal inputs and different outputs. The exact graph fails
  fresh case 64; no sound deterministic ONNX exists for all legal cases.
- **task340**: project interior occurrences of the four border colors to their
  corresponding inner border lanes and clear other interior markers. The exact
  cost-1173 graph passes 3000/3000 fresh and all structural gates. It contains
  zero `graph.value_info` entries; the metadata search had nothing to shave,
  and prior exact algebra/rebuild searches establish the current floor.
- **task365**: select the separated rectangle with the unique maximum red count
  and return its bounding-box crop. The exact cost-1381 graph passes 3000/3000
  fresh and all structural gates. Every one-dimension VI probe failed to yield
  a strict cheaper known-correct graph. The only repository-wide cheaper
  candidate, cost 1199, mismatches 441/3000 independent random cases.
- **task370**: repeat the black origin sprite along the hint-defined diagonal,
  preserving partial boundary copies. The exact cost-1011 graph fails fresh
  case 1883 by two cells. The sound 14-tap spec rebuild costs 2687 and cannot
  replace it under the strict-cheaper rule.

## Decision

No B3 artifact should be merged. In particular, do not use the visually large
nominal gains from task089 or task208: both are 100% runtime failures caused by
value-info allocator corruption, not real score improvements.
