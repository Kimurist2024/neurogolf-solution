# C9 strict SOUND rebuild report

## Outcome

No strict winner was found for tasks 310, 330, 340, 354, 361, 364, or 368
against the exact `submission_base_7999.13.zip` payload (SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`).
The accepted list is empty, projected gain is `+0.0`, and no root ZIP, CSV,
ledger, or promoted artifact was modified.

The task generators were read directly and compiled as these rules:

- task310: locate the unique complete 5--8 square perimeter and return its crop;
- task330: recolor each separated gray creature red iff it has six cells,
  otherwise blue;
- task340: project each interior occurrence of an outer-border color to its
  corresponding inner border lane and clear other interior markers;
- task354: use the three top lights to recolor the aligned gray rectangles;
- task361: complete all four rotations of the partial pinwheel;
- task364: classify the separated green glyphs as L/Y/H and recolor them
  blue/pink/red;
- task368: copy the colored prototype sprite into every gray copy.

## Decisions

| task | exact cost | strict evidence | decision |
|---:|---:|---|---|
| 310 | 566 | exact graph has no `value_info`, but fresh is 4993/5000 under `ORT_DISABLE_ALL` and 4998/5000 under default ORT; a new no-cloak safe selector is 5000/5000 on both runtimes with zero errors but costs 633 | reject safe reference on cost |
| 330 | 897 | declared `GroupNormalization` output `g` is 1x1x1x1 although the fixed input forces 1x10x30x30; lowest harvested non-base static floor is 923 | reject shape cloak / no cheaper history |
| 340 | 1173 | no `value_info`; 5000/5000 on both runtimes, zero errors; 84 historical rows bottom out at cost floor 1173; all 48 individual optimizer passes yield no cheaper known-correct graph | current floor, no winner |
| 354 | 537 | `gn_f` and `data_clear_f` are declared 1x1x1x1 although each `GroupNormalization` must be 1x10x30x30; best actual harvested alternative is 560 | reject shape cloak / cost |
| 361 | 968 | `gn` is declared 1x1x1x1 although runtime is 1x10x30x30; archived costs 810 and 854 fail complete known, while 1004 and 1006 are not cheaper | reject shape cloak / correctness |
| 364 | 685 | `input_fake` is declared 1x1x1x1 although `CenterCropPad(input, Shape(input))` must preserve 1x10x30x30; the known-correct true-rule rebuild costs 89186, and lower rebuilds are wrong | reject shape cloak / cost |
| 368 | 521 | fresh is 5000/5000 on both runtimes, but `gn` is declared 1x1x1x1 although `GroupNormalization` must be 1x10x30x30; all harvested alternatives have static floor at least 522 | reject shape cloak / cost |

The shape witnesses are terminal under the C9 policy even though ONNX checker
and strict inference accept the opaque `CenterCropPad` chains. A single honest
float32 `GroupNormalization` output at fixed shape 1x10x30x30 is 36,000 scored
bytes, already above every affected compact baseline. For task364 the same
36,000-byte lower bound follows from the first `CenterCropPad`, whose target is
exactly `Shape(input)`. Removing all supplied `value_info` and correcting the
output declarations makes all four de-cloaked probes unscorable by the official
static pipeline, confirming that their low score depends on the supplied shape
metadata/opaque inference path.

## New task310 sound reference

`task310_safe_linear_selector.onnx` replaces the incumbent frequency test with
positive row weights derived from the complete generator support. The weights
make every present non-frame color's weighted sum strictly larger than the
frame's; uint8 underflow maps absent colors to 255, then `ArgMin` and
`TfIdfVectorizer` produce the selected channel. The existing crop contraction
is retained.

The candidate has zero `graph.value_info`, no `CenterCropPad`, no nonstandard
domain, no banned/Sequence/nested/function/sparse construct, and no Conv-family
bias finding. Full checker and strict shape/data propagation pass. It is exact
on all 266 known cases and independently passes 5000/5000 with zero errors
under both default ORT and `ORT_DISABLE_ALL`. Its SHA-256 is
`ad81c541e1164adc0e6a7971ae62a013b79e3bd6a1a3efcbca93b4400084c496`.

It is nevertheless rejected because official-like cost is 633
(`memory=240`, `params=393`) versus 566. The score delta is
`-0.11187634394099177`. The exhaustive affine audit also proves that a cheaper
selector of the form `intercept + slope*row` cannot work: required slope ratios
simultaneously exceed `0.0555555555556` and stay below `-0.00609756097561`.

## Search coverage

- Existing scans contributed 23 task310, 3 task330, 84 task340, 5 task354,
  6 task361, 74 task364, and 4 task368 candidate rows.
- All 48 available single ONNX optimizer passes were run on the only two
  no-cloak baselines, task310 and task340. Each produced five unique serialized
  outputs including the base; no pass produced a cheaper known-correct model.
- All exact members pass full checker, strict shape inference, standard-domain,
  banned-op, nested-graph, function, sparse-initializer, and Conv-bias checks.
  These generic gates do not override the explicit operator-shape witnesses.
- No lookup memorization, shape/value-info reduction, host undefined behavior,
  or unsafe Conv bias was used or accepted.

Evidence: `baseline_anatomy.json`, `baseline_audit.json`,
`task310_safe_score.json`, `task310_safe_audit.json`, `structural_audit.json`,
`optimizer_sweep.json`, `history_summary.json`, `decloaked_audit.json`, and
`decloaked_runtime_audit.json`.
