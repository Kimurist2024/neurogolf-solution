# C12 archive and structural-floor SOUND report

## Outcome

No safe improvement was found for tasks 102, 124, 132, 163, 175, 178, or
228. The accepted manifest is empty, projected gain is `+0.0`, and no root
submission ZIP, CSV, score ledger, or shared handcrafted model was changed.

The immutable cost authority is `submission_base_7999.13.zip`, SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
All exact members were extracted directly from that archive and remeasured.

| task | SHA prefix | exact cost | memory + params | retained leads | decision |
|---:|---|---:|---:|---:|---|
| 102 | `48d974a3` | 493 | 381 + 112 | 8 | only cheaper lead is unsafe and fresh-wrong |
| 124 | `3703bae3` | 265 | 234 + 31 | 5 | one shave crashes; other leads tie/lose |
| 132 | `2ac5f792` | 316 | 0 + 316 | 8 | every cheaper factor deletion is known-wrong |
| 163 | `546f07f1` | 196 | 0 + 196 | 4 | every cheaper factor deletion is known-wrong |
| 175 | `0979ba89` | 166 | 0 + 166 | 4 | every cheaper factor deletion is known-wrong |
| 178 | `4d5c0440` | 269 | 208 + 61 | 2 | exact sound floor retained |
| 228 | `13263c21` | 302 | 252 + 50 | 8 | cheaper lead passes 0 known; exact lead only ties |

The 39 retained `lane_archive_top200` models cover the byte-distinct models
from the archive ZIP/loose sweeps and the older direct candidates cited by the
inventory. C12 completed isolated actual-cost and known-set audits for 38; the
remaining task124 r01 is itself a reproducible process crash. Every loadable
model passes ONNX full checking, strict shape inference with data propagation,
standard-domain, banned/sequence-op, nested-graph/function/sparse, and exact
Conv-family bias-length checks. Those static checks do not override runtime,
truthful-shape, exactness, or cost failures.

## True generator rules

- **task102 / `task_44d8ac46.py`:** fill the interior of complete square gray
  frames of side 4, 5, or 6 with red, while leaving rectangular, nested, and
  junk-frame interiors black.
- **task124 / `task_53b68214.py`:** infer a 1x1, 2x2, 2x3, or 3x3 sprite and
  continue its vertical or diagonal periodic placement from the partial
  height-5..8 input to a 10x10 output.
- **task132 / `task_56ff96f3.py`:** each color occurs at two opposite corners;
  fill the complete axis-aligned rectangle between them.
- **task163 / `task_6d0160f0.py`:** the yellow cell identifies one of nine 3x3
  mini-panels and a coordinate inside it; copy that panel's colored pattern to
  the corresponding panel selected by the yellow coordinate.
- **task175 / `task_73251a56.py`:** restore the five erased rectangles in the
  fixed 21x21 quotient-pattern grid, with modulus and phase inferred from the
  surrounding values.
- **task178 / `task_746b3537.py`:** detect whether the grid is row-constant or
  column-constant and run-length-compress the varying one-dimensional color
  sequence to 3..5 colors.
- **task228 / `task_952a094c.py`:** remove the four colored inner-corner
  markers of the hollow rectangle and place them at the four diagonally
  opposite outer corners.

These rules explain why the tiny one-node tensor networks for 132/163/175 and
the compact coordinate carrier for 228 cannot be treated as specification
proofs merely because they cover the archived examples.

## Candidate dispositions

### task102

All eight candidates pass 267/267 known with optimization disabled. Their
actual costs are 500, 510, **491**, 1391, 494, 522, 537, and 642. Only r03 is
cheaper than 493. It fails default ORT because `CenterCropPad` receives a
one-element shape for three axes, and 55 declared tensor shapes contradict
execution; a single known example materializes 169115 intermediate bytes while
only 381 are charged. The same r03 model previously scored 925/1000 fresh, and
the exact incumbent scored 2721/3000. It is an unsafe two-cost shave, not a
sound frame detector.

### task124

r01 removes an unused Split output, the only nominal one-byte shave. The
isolated C12 audit exits 139/SIGSEGV, matching both `quick_k5` exit -11 and the
earlier lane19 fresh=1 crash. r02 ties cost 265; r03/r04 cost 266 and r05 costs
277. All runnable models reproduce the incumbent's five false shapes and
267/267 default-ORT runtime failures. No runtime-safe improvement exists.

### tasks132, 163, and 175

Each exact incumbent is already one output-free Einsum, so cost is only its
initializer elements. The complete deletion/factor-sharing history was rerun:

- task132: costs 282x4 and 287x4; seven pass 0/267 known and one passes 14/267;
- task163: costs 136x2 and 188x2; all pass 0/267;
- task175: four cost-129 variants; all pass 0/266.

There is no node-output memory to optimize, and every attempted parameter
deletion destroys the known mapping before fresh validation.

### task178

r01's nominal static label is misleading: runtime profiling exposes cost 8355
and every one of 268 known examples raises an allocator shape error. r02 costs
261 but passes only 68/268. The cost-269 exact member is the earlier
specification-derived run compressor, already validated 5000/5000 in lane17;
the prior sparse-Conv reduction fails ONNX full shape inference, while dense
factorization adds a 30x30 intermediate.

### task228

r01 costs 294 but passes 0/266. r02 is complete-known under both modes but only
ties the exact cost 302 and retains four false 16-way coordinate-carrier shapes
(the exact archive member has eight). r03 and r05-r08 cost 305..340; r04 cannot
instantiate the uint8 TopK kernel. Every complete-known graph uses the same
TopK/Einsum/Scatter coordinate carrier and is shape-cloaked; none is a cheaper
candidate, let alone a sound implementation of the four-corner rule.

## Fresh-5000 gate

No candidate reached the prerequisite intersection of strictly lower actual
cost, complete known correctness under both `ORT_DISABLE_ALL` and default ORT
with zero errors, runtime-shape truthfulness, and structural/UB safety.
Therefore there is no eligible candidate on which an independent 5000+5000
fresh run could establish adoption. Fresh testing cannot rescue a known
failure, SIGSEGV, unsupported kernel, cost tie, or shape/value cloak, and C12
claims no adoption fresh result.

Evidence is in `candidate_audit.json`, `crash_reproduction.json`,
`rejected_manifest.json`, and the empty `winner_manifest.json`.
