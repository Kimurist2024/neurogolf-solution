# Lane C21 — exact 7999.13 audit for tasks 138 and 187

No candidate is promotable. The lane projects **+0.0** and leaves the exact
`submission_base_7999.13.zip`, score files, ledger, and handcrafted models
unchanged.

## Exact authority

- Archive SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- `task138.onnx`: SHA-256 `9fb018e474f80e3e60be93c263bdf764aa9bca084fffc61cfd4f0df8f905cdc9`, cost 2731
- `task187.onnx`: SHA-256 `20f103296641dacfbb5ff424b60cf3006c9af42dba6ada7e985576609a2100b0`, cost 1814

Both members were copied byte-for-byte from the exact ZIP authority. The
archive still contains 400 unique entries and passes `unzip -t`.

## Task 138

The current 2731 graph already passed two independent 5000-case streams in
both `ORT_DISABLE_ALL` and default ORT in lane B8: 10,000/10,000 per mode,
with zero wrong results and zero errors.

The two unique archive leads both pass all 266 stored examples in both ORT
modes, full ONNX checker, and strict shape inference, but their authoritative
scorer costs are 2762 and 2822. The filenames' older static estimates 2588 and
2648 are not authoritative. Neither can improve the score, so no redundant
fresh run was performed.

Prior task138 work already exhausted direct metadata, coordinate, dtype,
initializer-deduplication, and shape-fold paths. The apparent `Shape(qcol)`
fold exposes the real CenterCropPad channel dimension and fails the full
checker; accepting it would require the prohibited shape-cloak route.

## Task 187

Eight unique cheap archive leads were audited.

- Three int8-TopK graphs cannot create an ORT CPU session in either mode.
- Four executable graphs cost 1853, 1859, 1969, and 3233, all above the exact
  1814 baseline.
- The sole strict-cheaper executable graph costs 1737 and passes 266/266 known
  examples in both ORT modes. It has 14 declared/runtime shape mismatches and
  nine CenterCropPad nodes, so it violates the no-shape-cloak gate.

The 1737 graph was nevertheless subjected to the requested high-k
counterexample audit. With seed 1872101 it produced identical results in both
ORT modes: **4695/5000 correct, 305 wrong, zero inference errors (93.9%)**.
The first mismatch occurs at valid case 8 on a 23x23 input. It therefore also
fails the explicit 95% adoption threshold, independent of the structural
rejection.

The exact 1814 baseline itself is not a sound replacement source: retained
lane-B2 evidence is 2996/3000 fresh, and default ORT currently rejects its
TopK shape contract. Known sound-ish rectangle rebuilds are much more
expensive. No error-free, shape-clean, cheaper task187 candidate exists in the
audited pool.

## Gate summary

All candidates were checked for official-like cost, stored train/test/arc-gen
correctness in both ORT modes, full checker, strict shape inference with data
propagation, runtime-versus-declared shapes, standard domains, banned or nested
graphs, functions, sparse initializers, giant lookup/Einsum indicators, and
unsafe convolution bias. Only the one genuinely cheaper executable lead was
advanced to 5000 fresh cases in each ORT mode; it failed decisively.

Machine-readable evidence is in `candidate_audit.json`, the two
`fresh_task187_r07_*_5000.json` reports, `rejected_manifest.json`,
`winner_manifest.json`, and `validation/root_integrity.json`.
