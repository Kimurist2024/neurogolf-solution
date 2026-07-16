# Lane A7 — exact 7999.13 baseline

Targets: task174, task153, task325, task071, task055, task088, task086.

The immutable source was `submission_base_7999.13.zip`, SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
Only this lane directory was written.

## Result

No safe candidate is promotable. Lane gain is **0.0**.

- task071's exact Cast replacement removed one parameter but made the runtime
  extent profiler-visible: cost **235 -> 286**.
- task088's exact axis-free reductions removed four parameters but similarly
  exposed row/column extents: cost **230 -> 342**.
- task055's lower-rank candidates reached costs 195/210/225 but failed every
  sampled fresh case and bundled gold. The exact rank-6 control passed but cost
  **270**, above baseline **234**. A generator-constrained degree-4 refit is
  infeasible on 435,576 exact-carrier sign constraints; the looser degree-4
  candidate cost 223 but was gold/fresh incorrect. Exact sparse reuse of `X`
  would save 12 parameters, but strict ONNX shape inference rejects sparse
  input rank propagation into Einsum.
- task153 and task174 apparent optional/deduplicated inputs are forbidden by
  their ONNX schemas. task325 and task086 have no lower-cost standard carrier
  substitution that preserves their already-verified runtime allocation.

Because no candidate passed the cheaper-and-known-exact gates, fresh 5000 was
not run; doing so cannot make any candidate eligible.

Machine-readable details are in `audit_rejections.json` and
`winner_manifest.json`.
