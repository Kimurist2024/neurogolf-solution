# high153 — task044/task117/task330 exact deep audit

## Outcome

No admissible strict-lower candidate was found against immutable
`submission_base_8009.46.zip` (SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`).
Projected gain is `0.0`; `winner_manifest.json` is empty. No root submission,
score file, docs file, or `others/71407` artifact was changed by this lane.

| task | authority cost | terminal authority issue | cheapest truthful/rule control | decision |
|---:|---:|:---|---:|:---|
| 044 | 1076 | 2 false 1x1 declarations; generator non-injective | 1702 | no safe lower graph |
| 117 | 605 | strict inference fails, default ORT fails, 10 shape contradictions | 6762 | no safe lower graph |
| 330 | 896 | default ORT fails, 38 shape contradictions | 5525 | no safe lower graph |

There are no strict-lower candidate SHAs to report. Diagnostic probes that
failed before actual-cost admission remain only in `candidate_audit.json` and
`mechanical_audit.json`; they are not integration candidates.

## task044

The authority member costs 1076 and is 266/266 known-correct in both ORT modes,
but `gn` and `u` are declared `[1,1,1,1]` while both execute as
`[1,10,30,30]`. Runtime tracing exposes 45,872 intermediate bytes versus the
nominal 877 memory. It therefore fails the truthful-shape gate independently
of known accuracy.

The generator creates two gray containers, two translated colored copies of
the black interior creatures, and unrelated dust. Output colors each interior
hole from its matching external creature and erases the external copies. The
generator is non-injective: legal byte-identical inputs with different outputs
have been reproduced at seeds 1503 and 1506. Consequently, no deterministic
ONNX has a generator-entailed total tie rule.

The 43-node/23-initializer graph has no dead node, unused or duplicate full
initializer, no-op, CSE, optional output, constant fold, absorb, or schema-valid
initializer-to-attribute reduction. The retained autocorrelation control costs
1702, contains two giant Einsums, preserves the same two false declarations,
and encounters the non-injective fresh ambiguity. It is neither cleaner nor
cheaper.

## task117

The authority member costs 605 and is 265/265 under `ORT_DISABLE_ALL`, but
default ORT cannot create a session. Strict data-propagating inference fails,
runtime tracing finds 10 declared/actual contradictions, and actual
intermediates total 73,571 bytes versus nominal memory 496.

All 67 nodes and all 18 initializers are live; there is no duplicate full
initializer. Exact dead/CSE/no-op/optional/absorb scans are empty. The sole
mechanical semantic rewrite folds `Shape(slice_axes_chw)` to its fixed scalar
shape. That exposes the hidden vector rank and makes the existing
`CenterCropPad` declarations fail full/strict inference, so the model is
unscorable rather than lower.

The truthful generator-rule control reflects the leg sprite across both axes
around the X center. This run independently rechecked it: cost 6762,
265/265 known in both ORT modes, zero runtime shape mismatches, standard domain,
and no Conv-bias finding. It is 6157 cost above the authority member and cannot
enter the fresh gate.

## task330

The authority member costs 896 and is 266/266 only under `ORT_DISABLE_ALL`;
default ORT fails session creation. Runtime tracing finds 38 shape
contradictions and 83,177 actual intermediate bytes versus nominal memory 730.
The compact graph's score depends on a long opaque `CenterCropPad` chain and is
not an admissible parent for another exact regolf.

All 44 nodes and 9 initializers are live, with no duplicate, no-op, CSE,
optional-output, or absorb opportunity. Its six `ConstantOfShape([1])` values
are already stored as zero-parameter attributes. Folding them to one-element
initializers is formally exact, but it exposes the hidden target shapes;
checker, strict inference, and both ORT sessions then reject the graph. Thus
the nominal `-42` arithmetic cannot produce an actual-cost candidate.

The truthful component-size control implements the rule “red iff component
size equals six, otherwise blue.” It rechecks at cost 5525, 266/266 known in
both ORT modes, zero runtime shape mismatches, and Conv UB0. The historical
cost-807/808/817 pair-frame variants are not alternatives: they score only
166/266, 162/266, and 210/266 known, respectively.

## Gate conclusion

Fresh dual-ORT generation was not run because no graph passed the preceding
mandatory gates: actual strict-lower cost, full checker, strict and truthful
shape, both-ORT known completeness, and UB0. Fresh testing cannot rehabilitate
an unloadable graph or a control already thousands of cost above authority.
Sparse exploration was excluded as instructed.

Evidence is in `baseline_audit.json`, `mechanical_audit.json`,
`candidate_audit.json`, `control_audit.json`, `manifest.json`, and
`winner_manifest.json`.
