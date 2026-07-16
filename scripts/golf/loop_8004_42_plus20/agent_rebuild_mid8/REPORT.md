# Mid8 rebuild audit — tasks 036 / 208 / 255 / 044

Base authority: `submission_base_8004.50.zip` (SHA-256
`63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`).
No root ZIP, score CSV, best-score ledger, or shared handcrafted model was
modified.

## Outcome

One strict-cheaper candidate is admissible:

| task | current | candidate | gain | known dual | independent fresh dual | decision |
|---:|---:|---:|---:|---:|---:|---|
| 036 | 1477 | **1428** | **+0.0337381396** | 265/265 in both modes | 5000/5000 on each of two seeds in both modes | **accept** |
| 208 | 1422 | sound floor 1812 | — | sound control complete | prior 1000/1000 | no cheaper sound graph |
| 255 | 1162 | — | — | — | — | generator is non-injective |
| 044 | 1086 | — | — | — | — | exhaustive frontier has no cheaper graph |

The accepted payload is
`candidates/task036_truthful_gather.onnx`, SHA-256
`fc83bef42ce52ddd5c726323bacca5c4bf59ecaa55ef2aa55b1571243e9b5738`.
It measures as memory 1194 + params 234 = cost 1428.

## task036 evidence

The generator creates a connected special-color object with support bounded by
3..5 rows and columns, protects it with a one-cell moat, and scatters unrelated
colors outside that moat. The output is the special object's tight crop. The
candidate implements that rule with a fixed truthful 5x5 `GatherND` carrier.

- full checker and strict shape inference with data propagation: pass;
- all inferred shapes static and positive;
- declared/runtime shape mismatches: 0 over all 33 traced node outputs;
- traced intermediate bytes: 1194, exactly the official-like memory result;
- standard ONNX domains, no nested graph/function/sparse initializer/banned op;
- Conv bias findings: 0;
- no lookup table, no shape cloak, no giant initializer;
- three analytic `Einsum` nodes, maximum five inputs, below the campaign's
  giant-contraction cutoff of more than 16 inputs;
- both ORT modes: 265/265 known, zero runtime errors;
- seed 260714036: 5000/5000 in each ORT mode, zero runtime errors;
- seed 910714036: 5000/5000 in each ORT mode, zero runtime errors.

This candidate has no private-zero lineage. Full machine-readable evidence is
in `task036_audit.json`.

An additional attempt narrowed the dynamic crop indices from int64 to int32.
It would have saved 400 intermediate bytes, but opset-23 `GatherND` rejected
int32 indices during full checker shape inference. The invalid variant was not
serialized or admitted.

## task208

The true rule copies the colored frame around the first of two equal black
rectangles to the second rectangle. The current-history frontier contains three
nominally cheaper candidates, but all have nonstatic tensors and private-zero
lineage. The cost-1392 candidate is especially misleading: it produces 266/266
known runtime errors and 3000/3000 fresh runtime errors due to a `Slice`
allocator shape mismatch.

The existing spec-derived exact control
`scripts/golf/scratch_codex/task208/cand_current_exact_full_trim.onnx` passes
known/strict checks and prior fresh 1000/1000, but costs 1812, above the current
1422. Thus there is no strict-cheaper sound candidate, and no private-zero
candidate meets the required multiple-seed 100% guarantee.

## task255

The generator hides every green interior cell by converting it to black in the
input. A legal bottom-clipped vein with height 3 and one with height 4 can
therefore produce byte-identical inputs but outputs differing in 15 cells.
`scripts/golf/scratch_codex/task255/ambiguity_proof.py` reproduces the witness.
Consequently a deterministic ONNX cannot be exact on all legal instances, so a
private-zero-derived candidate cannot receive the requested pass guarantee.
No candidate from the 60-SHA current rescreen is admissible.

## task044

The generator places two gray containers, two translated colored creatures,
and unrelated dust; the output moves the creature colors into matching holes.
The exhaustive prior audit covered 112 unique SHA values. No model scored below
the old 1087 base, while the current member is already 1086. Independent latent
enumeration also found legal identical-input/different-output cases at seeds
1503 and 1506. There is no strict-cheaper candidate in the inventory.

`result.json` is the concise promotion manifest. Only task036 should be
considered for the parent's next no-regression ZIP build.
