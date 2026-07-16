# A17 strict history and task051 factor audit

## Outcome

- Exact source: `submission_base_7999.13.zip`
- Source SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Tasks: 029, 051, 195, 397, 400
- Archive retained candidates: 1
- Exact-byte-distinct loose historical models: 139
- Safe winners: 0
- Verified gain: `+0.000000`

No root ZIP, CSV, score pointer, or ledger was modified by this lane.

## Baseline and history result

| Task | Exact cost | Unique loose models | Result |
|---:|---:|---:|---|
| 029 | 270 | 32 | 30 alternatives are not cheaper; one is unscorable. The retained static-267 lead is that unscorable model. |
| 051 | 283 | 24 | 14 alternatives are not cheaper and nine are structurally rejected. |
| 195 | 150 | 25 | 23 alternatives are not cheaper and one is structurally rejected. |
| 397 | 364 | 36 | 32 alternatives are not cheaper, two are unscorable, and one is structurally rejected. |
| 400 | 164 | 22 | All 21 alternatives screen at or above baseline. |

The shared archive inventory covers 1,195 ZIPs, 224,111 ZIP members, and
118,938 loose observations. Its only retained lead for these tasks is
`task029_r01_static267.onnx`. Although its static floor is three units below
the exact member, the official-like runtime cannot create a session: a `Max`
node mixes `int64` and `int32`. It therefore fails complete-known admission.

## task051 exact factorization

The current model has two 4x3 initializers related exactly in float32:

`J2 = J1 @ [[1,0,0],[1,0,0],[0,0,1]]`

Replacing the 12-element `J2` with this 9-element transform is algebraically
exact, passes full checking and strict data-propagating shape inference, and
passes all known examples. The actual profile improves from 283 to 280
(parameters 203 to 200; memory remains 80).

It is not admissible. The transform merely replaces the two `J2` operands in
the final operation; the final Einsum still has 65 inputs, so the structural
gate returns `giant_einsum:65`. There is no existing 3x3 initializer, no
diagonal extraction of an existing 3x3x3 initializer, and no transformed
rank-3 initializer that supplies the transform for free. Absorbing it into the
only adjacent rank-3 carrier requires a new 27-element tensor while removing
only 12 elements, a net +15 parameters, and still leaves 65 operands.

The rejected probe is retained only as evidence in
`candidates/task051_j1_factor.onnx`; it is not a winner or submission member.

## Current-model safety audit

Tasks 029, 051, 195, and 400 pass full ONNX checking, strict shape inference
with data propagation, standard-domain inspection, and Conv-bias inspection.
They have no unused initializer or identical same-shape initializer pair.
task051 is the sole giant-Einsum graph among those four.

task397 passes full ONNX checking and known scoring, uses standard domains, and
is Conv-bias clean, but strict data-propagating inference fails at `Gather`
because an index is outside the inferred rank. This exact member came from the
fixed baseline; no derived task397 candidate was admitted, and every historical
alternative was cost-dominated, unscorable, or structurally rejected.

## Admission disposition

No model survived both the strictly-cheaper actual-cost test and the structural
safety gate. Consequently there is no candidate for independent fresh
5000/5000 validation, and `winners` is empty in `final_manifest.json`.
