# Deep69 — task349/138/076/107 generator-first rebuild audit

## Outcome

No candidate satisfies the required combination of strict-lower cost, complete
known correctness in both ORT modes, truthful runtime/static shapes, safe
structure, and generator/private guarantee. Accepted: **0**; projected gain:
**+0.0**. The immutable `submission_base_8005.16.zip` and every protected root
artifact were left unchanged.

The authoritative 8005.16 costs are task349 **3564**, task138 **2729**, task076
**2550**, and task107 **708**. This differs from stale `artifacts/optimized`
brief values and is why those briefs were not used as the comparison authority.

## True-rule conclusions

- **task349**: recover one to six maroon rectangles, infer radius from their
  `2r` run, paint downward blue beams and radius-dependent green halos, then
  restore maroon cores. The latest signature-free, six-object, truthful
  adjacent-overlap rewrite is `adjacent_affine_domain.onnx`: known **267/267**
  in both ORT modes, full checker/strict data propagation, zero runtime-shape
  mismatches, but cost **4022**. It misses the strict-lower gate by **458**.
  The more general exact subset-OR rewrite costs **4572**. No fresh run was
  spent on the 4022 graph after it failed the mandatory cost pre-gate.
- **task138**: detect the four colored border lines, crop the enclosed
  rectangle, and propagate pixels matching each border color toward that
  border. Both retained known-perfect alternatives reprice to **2762** and
  **2822**, above 2729. Earlier coordinate, dtype, initializer, metadata, and
  `Shape(qcol)` folding work is exhausted; the fold exposes the real
  CenterCropPad dimensions and fails the truthful-shape gate.
- **task076**: use the first complete colored sprite to restore blue/green
  pixels in rotated partial copies. The generator is non-injective: valid
  `megarotates=[0,3,3]` and `[0,1,1]` parameterizations produce the same input
  hash but outputs differing at 12 cells. Therefore no deterministic
  input-only ONNX can have the required full private guarantee. All five cheap
  archive leads also fail target ORT initialization on unsupported integer
  TopK.
- **task107**: infer scale `A=1+number_of_distinct_bottom_row_colors`, expand
  the 5x5 grid by `A`, and draw the red diagonals determined by the 2x2 box
  location. The cost-706 optional-input shave and cost-638 historical graph
  are known-perfect in both ORT modes, but retain respectively 11/13 runtime
  shape mismatches, 58/66-input giant Einsum, and GatherND. Both violate the
  explicit cloak/giant/lookup exclusion. A truthful ground-up expansion must
  represent the data-dependent scale/output map and did not beat cost708.

## Gate evidence

`evidence_summary.json` records immutable member hashes, full-known outcomes,
official-like runtime-trace costs, structural checks, and rejection reasons.
Supporting retained evidence is in:

- `root_high55/history_lead_audit.json`
- `loop_7999_13/lane_c22/generator_analysis.json`
- `loop_7999_13/lane_b8/task349_failure_case.json`
- `scratch_codex/task349/agent_spec_formula.md`
- `loop_7999_13/lane_rebuild_b2/individual_shave_task138.json`

No candidate ONNX was emitted because every generator-derived truthful rewrite
failed the strict-lower precondition, while every numeric-lower lead failed a
mandatory structural or determinism gate.
