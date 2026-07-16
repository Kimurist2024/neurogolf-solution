# Lane C38 — task398 exact-only audit

## Result

No winner. The authoritative task398 model remains unchanged at cost 350
(memory 144 + params 206), SHA-256
`741d07c3cd4fa9cfe363aeb30573cb97edda0881857abeb5ae096b77773018e4`.
Lane gain is **+0.0**.

The baseline independently passed:

- full ONNX checker and strict shape/data propagation;
- standard domains, no banned ops, no functions/nested graphs/sparse initializers;
- truthful runtime tracing: 0 declared/actual mismatches and 144 measured bytes;
- all 268 known cases with 0 wrong and 0 errors under ORT_DISABLE_ALL;
- all 268 known cases with 0 wrong and 0 errors under default ORT; and
- the moved external validator preflight and known-case check.

The existing 69-input final Einsum is grandfathered from the authoritative
baseline. This lane did not create or enlarge a giant Einsum.

## Exact reduction search

The structural audit found no exact duplicate initializer or node, removable
identity, or same-shape constant carrier. The final `input,T` contraction is
not the scalar `n` contraction: it retains color axis `o` as an output axis, so
the apparent T-column-sum CSE changes semantics.

Dense rank factorizations do not lower score:

- `T` is rank 9: 180 factor parameters versus 100 dense parameters.
- `V` is rank 2: 64 factor parameters versus 60 dense parameters.
- flattened `K` has rank 2 and a nominal 22-versus-24 parameter factorization,
  but using it ten times enlarges the final Einsum from 69 to 79 operands.
  Materializing `K` instead adds a 96-byte intermediate and is net worse.

Reconstructing `one`, `out_size`, `V`, or the common Q suffix adds scored
intermediates or final operands. Sparse initializers are excluded because the
repository records them as a proven grader error.

## Rejections

- The cost-347 Q4-to-D carrier passes known/generator cases but is not exact:
  the independent seed-80004604 external differential has 4 threshold
  mismatches in 500 executable cases. It is not eligible.
- All three archived cost-332 models are wrong on all 268 known cases under
  both ORT modes.

Because no cheaper arbitrary-input exact model survived the first gate, no new
fresh-5000 or external-500 candidate run was warranted. No model was promoted
and no shared submission or aggregate file was modified.

Primary evidence is in `final_audit.json`; the exhaustive structural details
are in `exact_structure_audit.json`.
