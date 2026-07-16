# Cost 100–200 gold-only lane report

Authority: `submission_base_8012.23.zip` / LB 8012.23.

Admission rule used in this lane:

- maintained private-zero/known-black tasks excluded;
- rejected 8013.52 tasks 012/110/161/175/188/355 excluded;
- `try_candidate.py` structural validation required;
- official train/test/arc-gen gold must be exact;
- margin/runtime/nonfinite/shape errors must all be clean;
- candidate must have a strictly lower official cost.

## Result

No candidate was admitted. Confirmed score gain from this lane is **0.0**.

The root authority files were not modified. At the end of the lane,
`submission.zip` and `submission_base_8012.23.zip` both had SHA-256
`720ebf75d826945250e3c7d7ea11780a950d8d3038546e9c7595503277a1189f`.

## Search coverage

1. Prior exhaustive loose/ZIP history was rechecked. Its only strict-lower
   gold-exact lead in this cost band was task343@172, already known LB-black and
   therefore excluded.
2. Existing latent-component candidates for tasks 060/163/232/304/315 were
   rerun against official gold; every one failed gold.
3. Optional ONNX defaults were scanned (ignored Resize ROI, zero Pad value,
   default Slice axes/steps, default uint8 quantization zero point, zero
   Conv/Gemm bias). No applicable strict-lower winner existed.
4. Static `Shape` constant folding found 16 tasks with opportunities, but every
   rewrite exposed stale/cloaked downstream shapes and failed the mandatory
   strict structural checker. None was admitted.
5. Direct `ConstantOfShape` folding found six tasks (091/122/151/162/265/369),
   but all six similarly exposed shape-cloak inconsistencies and failed the
   strict checker. None was admitted.

## Important rejections

- task071: the historical CastLike scalar shave removes one parameter. After
  repairing truthful shapes it passes all official gold and margin checks, but
  its real cost becomes 303 versus authority 188. It is not an improvement.
- task243: `Shape(r0) -> Reshape(r0)` appears to offer 147→110 statically, but
  replacing the fixed shape reveals incompatible declared/runtime shapes. The
  full checker rejects it, so it is not safe.

## Evidence

- `optional_default_evidence.json`
- `static_shape_evidence.json`
- `constant_of_shape_evidence.json`
- `task071_build.json`
- `build_task243.py` (the candidate is rejected by the full checker before an
  artifact/evidence file is emitted)

Only entries listed in an evidence file's `winners` array are admissible. All
three scan evidence files have `winner_count: 0`.
