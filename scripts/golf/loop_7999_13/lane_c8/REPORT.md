# C8 sound deep-dive report

Reference: exact `submission_base_7999.13.zip` only  
Reference SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`

## Result

No model is eligible for promotion. Projected gain is **0.0**.

The exact ZIP members are all bundled-known-complete, but none satisfies the
full requested safety gate:

| task | exact SHA-256 | cost (memory + params) | independent generator audit | default ORT | decision |
|---:|---|---:|---|---|---|
| 054 | `f025ec3ad7285b2ce2c0069a32e6a7717472124516c39a9f02c5bc1a27f7595f` | 2291 (2038 + 253) | 4959/5000, wrong 40, errors 1 | session creation fails | unsafe |
| 209 | `9d0c21971843863f7b48a3a69a5cafcb15919e9641c851861dd46841059aa5fd` | 2218 (1963 + 255) | disable-all 4804/5000; default 4807/5000; errors 0 | runs | unsound and generator-non-injective |
| 367 | `c8c2ebc5b47a6c5f99074dada87ecc879ceafcdd55f8705a3da59f62e46fe1b6` | 2229 (2010 + 219) | disable-all 5000/5000, errors 0 | session creation fails | rule-sound only with optimizations disabled |

The task054 runtime failure is a `ScatterElements` out-of-bounds index 43 on a
length-30 axis. The default task054 failure is a declared/inferred Concat
dimension conflict. The default task367 failure is a `CenterCropPad` axes/shape
rank conflict. Fixed seeds and full exception strings are in
`baseline_audit.json`.

All three exact members pass full ONNX checking and strict shape inference in
the official disabled-optimization scoring path, use only the standard ONNX
domain, have no nested graphs, and contain no banned ops. Those checks do not
make the optimizer-dependent shape declarations safe under default ORT.

## Generator rules

### task054 / `264363fd`

Type D/B global region reconstruction. A reference star outside the boxes
defines the marker-center, arm, optional 3x3-fill colors and whether horizontal
and/or vertical arms exist. Each marker inside every box is replaced with the
reference star; its arm color is extended across that marker's complete box row
and/or column. Background, box color, number and placement of markers, flip,
and transpose are input-dependent. Correct handling requires dynamic color-role
discovery and isolation of separated box runs.

### task209 / `8a004b2b`

Type D crop/reconstruction. Yellow corner pixels delimit the output rectangle.
The bottom sprite supplies the complete color pattern; the partially shown
magnified copy inside the rectangle supplies magnification and alignment. The
output is the cropped rectangle with every sprite cell magnified and restored.

The generator is non-injective. The checked witness in
`scripts/golf/scratch_codex/task209/ambiguity_proof.py` constructs two legal
parameter sets with byte-identical inputs but different outputs (`icol=5,
shows=[1,3]` versus `icol=3, shows=[2,4]`). Therefore no deterministic
input-only ONNX model can be exact for every legal generator draw. This is not
repairable by more parameters or a different architecture.

### task367 / `e73095fd`

Type B bounded geometry. Gray outlines form non-overlapping rectangles, with
optional gray connector lines between them or to a grid edge. Paint only each
rectangle's strict black interior yellow, preserving borders/connectors. Box
width/height are 3--7, there are 2--4 boxes, and horizontal placement can be
clipped at `col=-1` or the right edge. The compact rule engine therefore needs
validated row endpoints, clipped-edge handling and at most five rows of
vertical propagation; simple four-direction enclosure misclassifies connector
corridors.

## Exact-model cost anatomy

`baseline_anatomy.json` reproduces official costs by profiling all 266 bundled
examples and charging the maximum of declared and runtime shapes.

### task054

- Memory by dominant op: Concat 628 B, Einsum 396 B, ScatterElements 360 B,
  CastLike 132 B, Add 106 B.
- Largest runtime tensors: two 270 B `[1,9,30]` Concat banks, two 120 B
  `[1,4,30]` ScatterElements banks, and three 120 B `[1,2,30]` fp16 Einsums.
- Largest parameters: `P_src` 69, `termMap9x4` 36, the 30-entry row vector,
  coordinate source 26, `WR` 24, and center source 18.

These are the separated-run renderer and cannot be collapsed without dropping
the third marker/box or leaking across box gaps. The low-cost graph is already
fp16/uint8-heavy; the general FP16 pass only adds an input-boundary Cast.

### task209

- Memory by dominant op: Gather 414 B, Einsum 380 B, Cast 179 B, Div 124 B,
  Sub 101 B, Concat 88 B.
- Largest tensors: 360 B float `[3,30]` row-code Einsum, 300 B uint8
  `[1,10,1,30]` terminal Gather, 120 B int32 coordinate Cast, 80 B terminal
  Concat.
- Parameters are led by four 30-element coordinate/projection arrays, then a
  16-element row index and small QLinear kernels.

The only promising dtype probe converted six float initializers to fp16, but
full checking rejects it: `QLinearConv` requires `x_scale` float32, not
float16. No candidate was retained.

### task367

- Memory by dominant op: BitwiseAnd 520 B, Concat 240 B, Pad 240 B, Einsum
  160 B, MatMul 160 B, Sub 100 B.
- Largest tensors: 240 B row-event Concat, 240 B padded uint64 encoding,
  160 B uint64 MatMul, followed by eight 80 B int32 row states.
- The 180-element `selc` initializer is 82% of all 219 parameters; the next
  largest are `Pi` 20 and `CBi` 10.

The exact graph has 24 `CenterCropPad` shape cloaks. Removing them exposes the
true row-state sizes. The resulting honest graph is correct but costs 4445,
not 2229.

No exact member contains an unused initializer or a byte-identical duplicate
initializer, so constant deletion/sharing has no safe opportunity.

## Sound reference audits

The closest source-derived or shape-honest families were copied into
`sound_references/` and independently audited:

| task/model | cost | known | fresh disable-all | fresh default ORT | eligibility |
|---|---:|---:|---:|---:|---|
| 054 spec rebuild overdraw | 12380 | 266/266 | 5000/5000 | 5000/5000 | sound, but 10089 cost above exact |
| 209 fp16 full rebuild | 72842 | 266/266 | 4976/5000 | 4977/5000 | rejected; ambiguity remains |
| 367 bit-difference rebuild | 5285 | 266/266 | 5000/5000 | 5000/5000 | sound, but 3056 above exact |
| 367 no-shape-cloak graph | 4445 | 266/266 | 5000/5000 | 5000/5000 | closest honest model, but 2216 above exact |

All listed fresh runs have zero runtime errors. The no-shape-cloak task367
model contains zero `CenterCropPad` nodes.

## Full history search

The repository-wide `lane_rebuild_c2/scan_results.json` already deduplicated
and scored all available history for these tasks: 104 unique task054 models,
155 task209 models, and 69 task367 models. Against the exact 7999.13 members:

- task054 cheapest other known-correct model: cost 2372, 81 above 2291;
- task209 cheapest other known-correct model: cost 2290, 72 above 2218;
- task367 cheapest other known-correct model: cost 2247, 18 above 2229.

There is no scored historical model below the exact baseline for any target.
The nearest task054 and task367 families also fail default ORT with the same
shape-cloak errors. Historical shape/crop, dtype, arithmetic-fusion, axis,
constant-sharing and renderer mutations were reviewed; their failures agree
with the current anatomy:

- task054 direct spans leak between separated runs; compact role selection
  misses valid third markers; honest segmented reconstruction costs 12380.
- task209 reductions change decisions on generated inputs, and no deterministic
  model can solve the ambiguity witness.
- task367 removing row guards overfills connector corridors; removing a
  propagation stage misses height-7 interiors; replacing dynamic axes exposes
  true shapes; removing all cloaks raises cost to 4445.

## Acceptance decision

`winner_manifest.json` is intentionally empty. A score-only shave of the
current task054/task209 graphs would preserve their generator errors, and a
shave of task367 would preserve its default-ORT failure. Promoting any such
model would violate the explicit zero-error and soundness requirements.
