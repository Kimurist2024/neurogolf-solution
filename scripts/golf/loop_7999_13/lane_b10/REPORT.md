# B10 SOUND wave — exact 7999.13 baseline

## Outcome

No strict winner was found for tasks 123, 134, 143, 162, 169, 184, or
206. The accepted manifest is empty, projected gain is `+0.0`, and no root
submission ZIP, CSV, ledger, or shared handcrafted model was changed.

The only baseline authority used was `submission_base_7999.13.zip`, SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
Every baseline member was extracted from that exact archive and remeasured.

| task | exact cost | rule from raw generator | terminal result |
|---:|---:|---|---|
| 123 | 266 | extend the 5x5 periodic Chebyshev rings to 10x10 | sound floor retained |
| 134 | 423 | locate/downsample a scaled 3x3 megasprite and recolor it | cheaper truthful archive leads fail fresh-5000 |
| 143 | 212 | match the boxed creature and recolor the matching copy | only cheap lead catastrophically fails fresh-5000 |
| 162 | 451 | fill every empty 3x3 hole blue, in row-major order | CSE lead fails fresh and incumbent shape contract fails default ORT |
| 169 | 248 | recolor each gray component to `5 - component_size` | CSE lead has runtime errors; incumbent contract fails default ORT |
| 184 | 421 | recover the dominant color of each separated patch | best sound historical model costs 422; incumbent contract fails default ORT |
| 206 | 196 | copy the colored sprite around the gray marker | best validated historical model costs 199 |

## Long-distance candidate evidence

### task134

The incumbent is itself approximate: the earlier independent generator audit
found 2977/3000. It also has six declared/runtime shape mismatches. Of the
known-correct cheaper archive family, only r04 and r06 were shape-truthful and
safe to construct under both runtimes. Both failed new independent gates:

| candidate | cost | `ORT_DISABLE_ALL` | default ORT | errors |
|---|---:|---:|---:|---:|
| r04 | 320 | 4840/5000 | 4823/5000 | 0 / 0 |
| r06 | 322 | 4803/5000 | 4825/5000 | 0 / 0 |

The other known-correct r01/r05/r07/r08 candidates have six or seven false
declared shapes and are terminally ineligible. This is the Type-D global
scale/location/color inference failure called out by `SOUND_REBUILD_PROMPT`;
passing five random cases is not evidence of the full rule.

### task143

Archive r02 is truthful and costs 148, but independent fresh-5000 results are
only 2/5000 with optimizations disabled and 3/5000 with default ORT. The other
seven retained candidates fail known examples or even the five-case smoke
gate. The exact 212 graph has no dead nodes, duplicate initializers, or exact
duplicate expressions. Its 112 parameters are the two 30-element geometric
coordinate tables, one 20-element palette basis, and four dense 2x2x2
contraction tensors; generating those tables at runtime costs more counted
memory than it saves in parameters.

### task162 and task169

The retained historical algorithmic models are truthful but have real costs
of at least 828 and 710, versus 451 and 248. The aggressive task162 CSE probe
costs 373 but already misses 1/100 fresh cases. The task169 CSE probe raises a
Slice buffer-shape runtime exception on all 100 attempted cases. Independently,
the exact incumbents have 261 and 107 declared/runtime shape mismatches and
default ORT rejects their `CenterCropPad` shape contracts. No result derived
from those hidden shapes is eligible.

### task184 and task206

For task184, the two best known-correct archive models both cost 422, one
point above the exact 421 member; the exact member also has six false shapes
and cannot create a default-ORT session. For task206, archive r03/r04 pass
100/100 fresh but cost 199/200, above the exact 196. The exact task206 graph
has no duplicate initializer/expression shave, and its 30-element coordinate
initializer cannot be generated for less than the counted memory of the
generated coordinate tensor.

### task123

The exact 266 model is shape-truthful under both runtimes. The prior
independent audit is 3000/3000, and each of six shared CP-component deletion
probes scores 0/100. All eight retained static-166 archive candidates fail
known correctness and fresh 0/5. No safe parameter factorization beats the
12-memory/254-parameter incumbent.

## Structural gates

All exact members pass full ONNX checking and strict data-propagating shape
inference syntactically. Runtime tracing is stricter: mismatch counts are
0/6/0/261/107/6/12 for tasks 123/134/143/162/169/184/206. Exact task162,
task169, and task184 cannot construct a default-optimization ORT session.
Conv/QLinearConv bias checks are clean for the applicable task134 and task206
graphs. No nonstandard domain, function, sparse initializer, or nested graph
was accepted.

Machine-readable evidence is in `baseline_inventory.json`,
`exact_graph_audit.json`, `baseline_shape_safety.json`,
`task134_r04_fresh5000.json`, `task134_r06_fresh5000.json`, and
`winner_manifest.json`. Reused independent evidence is cited by path in the
manifest.
