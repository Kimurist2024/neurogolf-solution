# task158 current-8008.14 strict rebuild

## Outcome

One SOUND task158 replacement is accepted against the immutable
`submission_base_8008.14.zip` member. Actual cost decreases from **7578 to
7529** (memory 6709 -> 6662, params 869 -> 867), for projected score gain
**+0.0064870817284238035**. No ZIP, root score file, submission, or other lane
was changed.

- candidate: `sound/task158_exact_repair_cost7529.onnx`
- SHA-256: `9d9a3ca8fb39856125925ea464ed1cc80f0301bd785ff7b60da37bd1c2b6b9d1`
- authority member SHA-256: `2823587ecc3f1b5b158357b5c32638003130f133ba6ab64a35337238f134aead`
- trusted cost-7612 reference SHA-256: `3bfa73410f489f0bc444a1f4567f95837e445cd940d10b7282bdc50a95dd2dba`
- known: 266/266, wrong 0, errors 0 in both ORT modes
- fresh: 5000/5000 on each of seeds 1581081 and 1581082 in both ORT modes
- raw equivalence to the trusted cost-7612 graph: exact on all known and fresh
- raw safety: nonfinite 0, small-positive 0, min-positive 1.0, max-nonpositive 0.0

## Authority correction and exact repair

The 8008.14 task158 member SHA is byte-identical to deep46's rejected
`task158_scatter_max_invalid_zero_pruned.onnx`. It is public-corpus correct but
not SOUND: its invalid object slots have zero updates yet still carry arbitrary
ScatterElements indices. Deep46 fresh cases produced out-of-range indices 659
and 674 for a 650-element seed.

The trusted reference is deep46's accepted
`task158_scatter_max_orientation_only.onnx` (cost 7612, SHA above). It removed
only the two identity orientation gathers while retaining the ordered
top/left/magnitude tensors, so invalid slots stayed in range; it passed known
266 and two fresh seeds of 3000 in both ORT modes. By contrast, the authority
7578 graph also removed the three spatial/magnitude gathers and therefore made
invalid-slot indices arbitrary. The new 7529 graph keeps the authority's
cheaper valid-slot path, adds the explicit -1 invalid-base repair, and is raw
bit-identical to the trusted reference on all 10,266 audited cases per mode.

The winner keeps all valid slots unchanged and forces only an invalid object's
base index to float16 minus one. Anchor rows are in 0..25, invalid magnitudes
are in 0..12.5, and the flattened unsigned local-offset LUT is in 0..52, so
every invalid index is now in -1..649, entirely inside ScatterElements' accepted
inclusive range -650..649. Its update remains zero, and
`ScatterElements(reduction=max)` on a zero seed makes that update inert.

Five small integer/half-integer affine pairs were also fused into standard
variadic `Sum` nodes, and `(i << 1) | i` became `3*i` for i in {0,1,2}. All
partial values remain on the exact float16 lattice below magnitude 1024. These
rewrites are bit-exact for every reachable input, not empirical approximations.

## Audit

The winner passes full checker, strict shape inference with data propagation,
the shared structure gate, and runtime declared/actual shape tracing with zero
mismatches. It uses standard ONNX only, has no banned op, nested graph,
function, external/sparse initializer, lookup red flag, giant initializer,
giant Einsum, or Conv-bias issue. Maximum Einsum arity is three.

The reused complete history inventory contains 60 unique graphs (56 profiled);
its actual-cost floor was 7615. The current 7578 SHA was a later experimental
deep46 graph already rejected by fresh testing. Optional TopK Values removal is
invalid because that output is schema-required, and the current graph had no
duplicate typed initializer or unused initializer to CSE/prune.

Rejected controls are retained in `candidates/`: the repair-only model is SOUND
but costs 7584; the bool-Where source-high shaves fail ORT kernel loading; the
cost-7552 repair passes but is dominated by the accepted 7529 graph.

Authoritative evidence:

- `evidence/audit.json`
- `evidence/fresh_dual_affine_2x5000.json`
- `evidence/exact_proof.json`
- `evidence/history_reuse.json`
- `result.json`
- `winner_manifest.json`
