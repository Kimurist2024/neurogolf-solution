
===== BAND-SPECIFIC GUIDANCE (cost 51-150 -> goal <=50, micro-golf band) =====
Unlike the >5000 band, the incumbent here is ALREADY a micro net. BOTH approaches
are allowed and effective in this band:
  (a) aggressive representation-shaving of the incumbent (fewer/smaller
      initializers, cheaper dtypes, fused ops, dropped helper tensors), and
  (b) ground-up rebuild from the generator SPEC.
Read the incumbent first (artifacts/handcrafted/taskXXX.onnx if present, else the
seed in your scratch dir): in this band 1 saved param at cost~60 is worth ~+0.017
pts and halving 100->50 is worth +0.69 pts.

COST MODEL (exact): cost = params + memory footprint, where memory is computed by
STATIC shape inference over every value_info tensor (bytes = numel * dtype size).
Every intermediate tensor you materialize counts. Consequences:
  - Crop to the ROI as early as possible; keep intermediates tiny.
  - A terminal fused Einsum can reach memory=0 (task356-style, cost 94 today).
  - Chained diff-1 CenterCropPad stages (each pad_before=0, no shift) shrink a
    crop/cloak pipeline below the naive 29-base cost.
  - f16-casting the one-hot input is a TRAP (adds cost); f32 crop is optimal.
  - Use the smallest initializer dtypes that work (int8/bool); avoid f32 tables.

!!! PRIVATE-SET OVERFIT WARNING (this band's #1 failure mode) !!!
LB-verified incidents: ultra-cheap Gather/reflection nets that passed fresh
k=30 AND k=3000 AND all visible examples still scored ZERO on the private set
(e.g. a cost-30 fixed-reflection Gather later found failing arc-gen 4/267; a
whole wave of cheap 628-02 nets was private-0). Therefore:
  - Derive the rule from the GENERATOR CODE (task_<hash>.py), never from the
    visible grids. If your net encodes example-specific coordinates, sizes,
    color tables, or a fixed permutation that the generator actually samples
    randomly, it WILL be private-0 and lose ~21 pts instead of gaining ~1.
  - Verify EXACT on ALL train+test+arc-gen pairs AND >=1000 fresh generator
    instances before promoting. If the generator has modes/branches, make sure
    your fresh sample exercises every branch.
  - A sound cost-70 net beats an unsound cost-40 net. When a sub-50 form
    requires assuming anything not guaranteed by the generator, stop at the
    sound floor instead.

CORRECTNESS FOOTGUNS (all LB-verified on this project):
  - Conv/ConvTranspose bias length MUST equal the output channel count
    (short bias = ORT undefined behavior = random private flips).
  - sparse_initializer -> grader error. Loop/Scan/NonZero/Unique/Compress/
    Sequence/nested graphs/dynamic shapes -> reject. TopK only if unavoidable.
  - All tensor shapes must be static; final output [1,10,30,30] one-hot with
    out-of-bounds cells all-zero.
