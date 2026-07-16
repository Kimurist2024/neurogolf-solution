# Lane B15 — task023/task036 sound rebuild audit

Base authority: `submission_base_7999.13.zip`. This lane did not modify any
root ZIP, CSV, score ledger, or shared handcrafted model.

## Outcome

- Winners: **0**; projected aggregate gain: **+0.00**.
- Exact measured costs: task023 **1622 = 1245 memory + 377 params**;
  task036 **325 = 255 memory + 70 params**.
- Known dual ORT: task023 266/266 in both modes; task036 265/265 in both
  modes. Runtime/session errors: 0.

## task023

The generator places latent 2x2 boxes and length-3 sticks into an unlabelled
gray union. Seeds 29685 and 120072 reproduce the same 9x10 input but different
legal outputs. Therefore no deterministic input-only ONNX can implement the
full generator relation.

The exact cost-1622 graph scored 4195/5000 (83.90%) in each ORT mode on seed
150799913, errors 0. The parent-excluded cost-1497 family is not reused; its
existing independent evidence is 13/5000 in both modes. The other two archive
models cost 1520 and 1541. They pass all 266 known cases in both modes, but use
ArgMax/GatherND/Scatter or TopK/Scatter lookup/rank pipelines; the 1520 model is
also an explicit PRIVATE0 artifact. The 1541 model independently scored only
4389/5000 (87.78%) in both modes, errors 0. All are rejected before promotion.

## task036

The generator rule is to identify the compact connected special-color object
and return the complete input crop at its tight bounding box. The direct numpy
rule passed 265/265 fixtures and independent fresh 5000/5000, errors 0.

The exact cost-325 graph is not structurally admissible: a runtime trace finds
14 declared/runtime shape contradictions and 20,329 truthful intermediate
bytes on one known example versus the reported 255 memory. On independent
fresh seed 150799913 it scored 4978/5000 in both modes; all 22 failures were
output-shape violations (60x60 instead of 30x30), with no ORT exception.

Archive static floors 212, 214, 230, 231, and 232 do not survive real scoring:
their actual costs/correctness are respectively 1457/wrong, 259/wrong,
57371/wrong, 348/correct, and 68469/wrong. Every one also uses the same
CenterCropPad shape-cloak family; the 212/214 models additionally use a
17-input giant Einsum.

As a ground-up control, `candidate_task036_truthful_gather.onnx` replaces the
variable Slice/pad carrier with a fixed 5x5 int64 GatherND and a validity mask.
It has fully static truthful shapes, standard domain, no giant Einsum, no UB,
no lookup table, no shape cloak, and real cost **1428 = 1194 + 234**. It passes
265/265 known examples in both modes and independent fresh 5000/5000 in both
modes, margin 1.0, runtime/session errors 0. It proves the rule can be expressed
safely, but costs 1103 more than the exact base, so it is not a winner.

## Decision

No model reaches the intersection of strict lower real cost, generator-rule
soundness, complete known dual correctness, fresh5000 dual correctness, and
the no-cloak/no-lookup/no-UB structure contract. `winner_manifest.json` is
therefore empty.
