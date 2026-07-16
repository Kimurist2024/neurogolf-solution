# Lane C14 — cost 150–550 sound audit

## Result

No candidate is admissible. Lane C14 contributes **+0.0**, and the exact
`submission_base_7999.13.zip` remains unchanged.

- baseline SHA-256:
  `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`;
- tasks: 069, 071, 079, 091, 099, 105, 109;
- retained archive candidates audited: **41**;
- alternative/control models audited: **12**;
- accepted candidates: **0**;
- root ZIP, CSV, score ledger, and shared model artifacts: **not modified**.

Every loadable model was remeasured with the same actual scorer path and
tested on the complete known corpus under `ORT_DISABLE_ALL` and default ORT.
The audit also ran full ONNX checking, strict shape/data propagation,
standard-domain/banned/function/nested/sparse checks, Conv-family bias checks,
and runtime tracing of every intermediate. Per-model evidence is in
`candidate_audit.json`.

## Summary

| task | exact cost | retained actual costs | decisive result |
|---:|---:|---|---|
| 069 | 541 | 541, 541, 548, 550, 554, 556, 584, 575 | none cheaper; equal-cost lineages have 78 false shapes, fail default ORT, and prior fresh is 99/100 |
| 071 | 235 | 186, 186, 187, 213, 196, 286 | all cheaper models are incomplete or runtime-dependent; fully truthful control costs 7780 |
| 079 | 210 | 209, 209, 209, 212, 229, 217 | only cheaper known-complete r02 is a shape-cloaked lookup carrier and fresh is 4962/5000 |
| 091 | 265 | 262, unscored, 251, unscored, 266, 267, 270, 290 | measurable cheaper models are known-wrong; complete models start at 266 and inherited compact graphs have false shapes |
| 099 | 398 | 397, 397, 397, 397 | every one-parameter factor reduction fails known gold |
| 105 | 199 | 198 | known-complete, but dual-runtime fresh is only 4980/5000 and the graph keeps 45-input Einsums |
| 109 | 406 | unscored, 422, 405, 406, 420, 425, 431, 444 | cost-405 is a value-info-only shave that preserves ten shape contradictions; truthful rebuild costs 669 |

## Task findings

### task069

The rule copies the multicolor exemplar into three cyan sprite locations and
erases the exemplar. The exact cost-541 graph is 264/264 only with
optimizations disabled; default ORT rejects its `CenterCropPad` shape input.
Runtime tracing finds 78 false declarations and 198520 bytes of physical
intermediates versus 478 charged bytes. Its prior fresh screen was 99/100.

None of the eight retained candidates is actually cheaper: the two apparent
static wins both remeasure at cost 541, and the rest are 548–584. A separate
spec-derived search was also inspected. The compact cost-754 result still
hides two 1x10x30x30 tensors behind scalar declarations. The honest v1/v2/v3
families have zero runtime/static contradictions but cost 7896, 7786, and
7498. Therefore no sound sub-541 representation exists in the explored
families.

### task071

The rule reconstructs a horizontally symmetric sprite from the unoccluded
half after a four-column rectangle overwrites the other half. The exact
cost-235 graph is complete on known cases, but declares output width 17 while
executing width 30 and has three additional false helper shapes. Its compact
carrier also contains four giant Einsums.

The cost-186/187/196/213 candidates either miss a known case or error on all
cases in one runtime. The only complete retained model costs 286. The
independent spec-derived, fully truthful v20 control has zero runtime/static
contradictions and prior 10000/10000 fresh evidence, but costs 7780. This
confirms that removing the cloak is not a cost improvement.

### task079

The rule chooses the most frequent repeated 3x3 sprite type and emits that
sprite. The cost-209 r02 model is 266/266 on known cases under both runtimes,
but it is terminally unsafe: five false shapes hide three 1x10x30x30 carriers
and two ten-way vectors; one example executes 63137 intermediate bytes while
only 144 are charged. It also uses `Hardmax` and a 17-input `Einsum`, and its
prior independent fresh result is only 4962/5000. The other two cost-209
models are wrong on 93/266 known cases; all complete alternatives cost more
than the exact 210.

### task091

The rule crops the rectangle bounded by the two gray glowstick columns. The
exact cost-265 graph is generator-derived and has prior exact fresh evidence,
but its low cost relies on 15 false shapes; one known execution materializes
74098 intermediate bytes versus 249 charged. The measurable cost-251/262
leads are known-wrong. The cost-266/267/270/290 leads are complete but
dominated. Prior alternative renderer work found exact CenterCropPad/Pad
families at costs 576 and above even before replacing the inherited compact
front-end shape carrier. There is no truthful cheaper path in this family.

### task099

The task fills each outlined object according to its embedded color marker.
The exact cost-398 finite-state decoder is structurally truthful on runtime
shapes. All four retained rank reductions save one parameter but fail the
complete known corpus. A broader 29-model historical scan found 25
known-correct models and four cheaper models, with zero overlap. The separate
exact-carrier deletion and duplicate-carrier fusion probes reached cost 383
but failed visible gold. The 310-parameter coefficient bank remains the exact
floor of the explored decoder family.

### task105

The generator creates a rectangular skeleton with an optional interior
horizontal or vertical cutline, then hides skeleton cells. When too little of
the cutline remains visible, its orientation is information-theoretically
ambiguous from the input. The executable cost-198 edit removes one initializer
and retargets six singleton Einsums. It is 266/266 under both runtimes, but the
independent audit is only 4980/5000 in each mode. It also retains nine giant
Einsums, with up to 45 inputs. It cannot pass the exact fresh gate.

### task109

The rule mirrors the top-left sprite across both axes around the central cross
and recolors it with the cross color. The cost-405 candidate is not an
executable optimization: clearing `value_info` makes it identical to the
cost-406 exact graph. Both contain ten runtime/static contradictions, including
declared output 1x10x19x30 versus actual 1x10x30x30, and physical
intermediates of 63587 bytes versus 362–363 charged. The algebraic
`GlobalLpPool` rewrite costs 422 and retains the contradictions. A separately
built dense/static rule engine has zero contradictions and is correct under
both runtimes, but costs 669. Thus the one-point apparent gain cannot be
accepted under the no-cloak policy.

## Fresh gate

The final gate requires a strictly cheaper, known-complete, structurally
truthful graph before running independent seed 5000 under each ORT mode. No
candidate met those prerequisites. C14 therefore reused decisive prior fresh
evidence for task069, task079, task105, and task109 and did not spend new
dual-5000 runs on already-disqualified models. `fresh_evidence.json` records
those results.

## Artifacts

- `candidate_audit.json`: 7 bases, 41 retained candidates, and 12 controls;
- `truthful_controls.json`: alternative no-cloak/dominated family evidence;
- `fresh_evidence.json`: prior exact fresh evidence and gate decisions;
- `rejected_manifest.json`: task-level rejections;
- `winner_manifest.json`: empty acceptance manifest;
- `audit_candidates.py`: reproducible C14 auditor.
