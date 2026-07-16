# lane_b8 — exact 7999.13 deep audit for tasks 080, 138, 349

## Outcome

No candidate satisfies every standing gate while being strictly cheaper than
the exact `submission_base_7999.13.zip` member. The lane emits an empty
`winner_manifest.json`, projects score gain `+0.0`, and does not write a root
submission, score CSV, ledger, or handcrafted artifact.

| task | exact cost | best cheaper attempt | strict result |
|---:|---:|---:|---|
| 080 | 3051 | none | baseline sound; structural floor |
| 138 | 2731 | Shape fold looked locally cheaper | full checker rejects exposed CenterCropPad shape conflict |
| 349 | 3964 | 3954 | fresh generator failure; reject |

## Baseline soundness

Tasks 080 and 138 were each tested on two independent 5,000-valid-case streams.
Every stream was run once with `ORT_DISABLE_ALL` and once with default ORT
optimizations. Both tasks achieved 10,000/10,000 on each runtime mode, with
zero inference/generator errors. Task080's generator can emit grids over 30;
those were skipped before counting valid cases exactly as the scoring contract
does.

Task349 did not survive the enlarged fresh audit. Seed `349101`, valid case
421, fails identically in:

- the exact 3964 ZIP member;
- the 3956 table-crop candidate;
- the 3954 table/relation candidate; and
- the prior 4861 model previously labeled sound after only 3,001 fresh cases.

The complete fixture and all 12 differing output entries are recorded in
`task349_failure_case.json`.

## Task349 findings

The generator proves that a maroon core radius is 1..5. Its normalized run
masks are `3, 15, 63, 255, 1023`, whose modulo-11 codes are exactly
`{0,2,3,4,8}`. Cropping four length-11 lookup tables to length 9 is therefore a
true generator-domain reduction. It gives cost 3956, known 267/267, margin 2.0,
full checker/strict inference, standard domain, zero functions/sparse
initializers, and default-ORT known correctness. The exact relation
`top_offset = hstart_offset + hend_offset - 1` lowers the known-correct cost to
3954.

Neither is adoptable because both inherit an unrelated incumbent rule defect.
For the failing case, `patch_sumR=495564` activates hardcoded diff updates
`-24576` at row 9 and `+24576` at row 12. That signature only describes the
sum of horizontal component masks and is not sufficient to decide whether the
vertical halo overlap needs correction. Here it is a false positive and erases
a 3x2 green halo. The optimization candidates correctly preserve incumbent
semantics, but incumbent semantics are not generator-complete.

## Task080 findings

The exact graph is already a compact generator-rule compiler. Its principal
counted tensors are the final 30x30 uint8 label Gather (900), float cell decode
(400), row Gather (360), line table/pads (265), selected int32 index vector
(120), and required 10x10 cell/center tensors. The two unused MaxPool value
outputs cannot be omitted because output 0 is mandatory. Direct index
generation, combined two-axis Gather, dtype changes, and removal of the
restricted 4x4 branch either exceed the current cost floor or have prior
generator/gold counterexamples. The exact baseline itself passed the enlarged
fresh gate.

This agrees with the existing broad `lane_headroom` pool scan and the detailed
task080 branch/failure history: no stored cheaper visible-correct candidate is
a sound replacement for cost 3051.

## Task138 findings

The exact graph's main tail floor is two uint8 role tables (210 each) followed
by two channel Gathers (300 each). Coordinates and role maps are already at
their useful dtype/shape floors; turning parameter tables into computed tensors
adds at least as much counted memory.

`Shape(qcol)` looked removable: its effective one-axis target value is `[1]`.
The attempt was intentionally made without adding a shape cloak. Full checker
then inferred the real CenterCropPad channel dimension 10, conflicting with the
incumbent-declared hidden dimension 1. Keeping the hidden declaration is not a
valid full-check model after the constant fold; removing it exposes much larger
tensors and cannot beat 2731. The attempt is therefore rejected at the checker
gate. Existing archive candidates bottom out above the exact baseline (the
best audited older member costs 2762), and the prior individual shape-shave
scan found no valid reduction.

## Gate discipline and files

The `neurogolf-onnx-golf` workflow was used: generator source first, then exact
ZIP-member profiling/history, full checker and strict inference, official-like
cost and known correctness, margin, independent fresh streams, and default ORT.
No sparse initializer, function, nested graph, foreign domain, new dynamic
shape, or new shape cloak was accepted.

- `build_and_known_report.json`: task349 table-crop cost/known/structure proof
- `offset_relation_report.json`: task349 relation variant cost proof
- `task349_failure_case.json`: deterministic fresh counterexample and internals
- `fresh_*.json`: independent mode/seed reports
- `task138_shape_fold_failure.json`: rejected checker experiment
- `failure_manifest.json`: machine-readable final decisions
- `winner_manifest.json`: empty approved set
