# Root exact audit 220 — tasks 222/341/357 and adjacent untouched targets

## Result

No candidate from this root-side audit is eligible for staging.  The active
`others/71407` set was changed only by the independently reviewed task158
winner from lane 215/219; none of the files in this directory was promoted.

## task341 identity-crop finding

The task341 authority contains `Shape(e186) -> [3]` followed by three
`CenterCropPad` operations whose data inputs are already length-three
initializers.  Redirecting their consumers to `e186`, `e199`, and
`target_shape_src_i64` and deleting those four nodes is an exact value
identity.

It is not a score improvement under truthful shapes.  The authority declares
large runtime tensors as `[1,1,1,1]`; its official profile is cost 260 even
though the output really is `[1,10,30,30]`.  The exact rewrite exposes the
constant Slice shapes.  After removing the stale declarations and recording
all runtime-truthful shapes, the candidate remains known-correct but profiles
at memory 127368 + params 29 = cost 127397.  It is therefore rejected and is
not an allowed shape-cloak-derived candidate.

- authority SHA-256: `03775b8b7d7a79549816cb66b328bc21a5b8b3d5713e23e1ce100b40afbce6c1`
- diagnostic truthful candidate SHA-256:
  `869f62b7c7752acb61fca74f22c6d4cec5cc3f45e040b28d45f4240a9bbd6ac1`
- builder: `build_identity_crops.py`

## Other inspected models

- task357 is already below the earlier canonical floor.  Its width-dependent
  bounce renderer needs both the 10-value row path and 16-value guarded column
  code; prior exhaustive direct-convolution and guard-tail searches found no
  correct strict-lower graph.
- task345's six negative-pad Conv row probes have a truthful nonnegative-pad
  control only at equal cost; the retained lower history is known-wrong.
- task275 is a compact 41-input Kronecker contraction.  Its four coordinate
  factors are distinct binary tensors and its retained lower polynomial is
  known-wrong; no dead or duplicate initializer exists.
- task222 was handed to a dedicated private-safe lane because its giant
  contraction is a highest-risk private-zero lineage.  It may be admitted only
  with an all-input algebraic identity or complete generator-support proof.

Root `submission.zip`, `all_scores.csv`, and the authority ZIP were not changed
by this audit.
