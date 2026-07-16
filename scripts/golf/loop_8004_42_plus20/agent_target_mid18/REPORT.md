# Target mid18 — eight-task strict audit

## Outcome

**No safe strictly cheaper model exists in the audited current/history/local
families. Accepted models: 0/8; cost delta: 0; projected gain: +0.0.** No ZIP,
root score file, or shared artifact was modified.

The original authority was `submission_base_8004.50.zip` (SHA-256
`63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`).
The final rebase authority is `submission_base_8005.16.zip` (SHA-256
`73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`).
All eight target member hashes are byte-identical between the two archives, so
the rejection result is rebase-compatible.

All eight tasks are absent from the 51-task private-zero/unsound catalog. No
candidate reached promotion, so no private-lineage claim is made.

## Compiled generator rules

- **task099 (`444801d8`)**: each width-five blue frame contains one non-blue
  seed; retain the frame and fill its interior plus open-side cap with that
  seed color.
- **task279 (`b2862040`)**: a blue connected component is a closed box exactly
  when its four-neighbor graph contains a cycle; recolor that entire component,
  including barnacles, cyan. Open components remain blue.
- **task345 (`d9f24cd1`)**: extend every bottom red seed upward; if gray blocks
  the next cell, step right and continue upward.
- **task239 (`9af7a82c`)**: count each color, order distinct counts descending,
  and render top-aligned vertical bars in the corresponding colors.
- **task075 (`363442ee`)**: copy the left 3x3 tile into each right-hand 3x3
  block whose center contains a blue marker; retain the gray separator.
- **task392 (`f8c80d96`)**: infer the edge center and thickness from the visible
  clipped concentric-mat prefix, then continue all rings on gray background.
- **task387 (`f35d900a`)**: infer the rectangle from four colored corners, draw
  the opposite-color 3x3 corner decorations, and add alternating gray edge
  marks.
- **task225 (`93b581b8`)**: copy the source 2x2 colors to four clipped 2x2
  blocks at the four diagonal offsets.

`reference_audit.py` implements those rules independently of every ONNX. Each
rule matches all stored known cases and two independent fresh streams:

| task | known | fresh seed A | fresh seed B | errors |
|---:|---:|---:|---:|---:|
| 099 | 265/265 | 5000/5000 | 5000/5000 | 0 |
| 279 | 266/266 | 5000/5000 | 5000/5000 | 0 |
| 345 | 264/264 | 5000/5000 | 5000/5000 | 0 |
| 239 | 267/267 | 5000/5000 | 5000/5000 | 0 |
| 075 | 265/265 | 5000/5000 | 5000/5000 | 0 |
| 392 | 266/266 | 5000/5000 | 5000/5000 | 0 |
| 387 | 266/266 | 5000/5000 | 5000/5000 | 0 |
| 225 | 265/265 | 5000/5000 | 5000/5000 | 0 |

The actual per-task seeds and counters are in `reference_audit.json`.

## Exact 8005.16 members and disposition

| task | SHA-256 | cost | known dual ORT | structural finding | decision |
|---:|---|---:|---:|---|---|
| 099 | `de18296f07fc021360fa0fec22861b284840aa44d939bfc14be85038dcc5d998` | 398 | 265/265 both | four prohibited giant `Einsum` nodes; largest has 34 inputs | reject lineage |
| 279 | `d3bb22792a3e44e09d21971f88642622a32d161f3a7888f9d0e9efe5862d0a9b` | 397 | 266/266 both | 203 declared/runtime shape contradictions; 385 charged bytes vs 615,924 profiled bytes | reject cloak |
| 345 | `36b1e2be6488496ca637996552b505cf9b2742775b663c4210407d61b861d636` | 389 | 264/264 both | six Conv nodes use negative pads | reject UB-like construction |
| 239 | `e15519d37ccaa4f3ad478091a2eb5a6a1fe984bc602a2ffd347f03c250eb68e0` | 384 | 267/267 both | clean: truthful shapes, standard domain, no giant/lookup/bias issue | retain incumbent |
| 075 | `ea2280d32f09e571182c0dbae57155a7e2b8a23a88d0a027ae9add3c9770ceb8` | 345 | 265/265 disabled; default load failure | 33 shape contradictions; 311 charged bytes vs 70,696 profiled bytes | reject cloak |
| 392 | `e68e69908e85933a9cc1367d8c8653560ed5b45188f0da3729a74e8d92491fb5` | 345 | 266/266 both | three prohibited `TfIdfVectorizer` lookup nodes | reject lookup |
| 387 | `df39eba7bd5bb4a6fad8c97f0f5e05ae0392a4990e829cc4435ff05220324e28` | 337 | 266/266 both | 16 shape contradictions; 242 charged bytes vs 90,657 profiled bytes | reject cloak |
| 225 | `c55b5673a1e36b07114e82a629d23b01cefbd7b56289ad314b272d7180ef8a4a` | 333 | 265/265 both | clean; profiled bytes equal charged 233 bytes | retain incumbent |

Every exact member passes full checker, strict shape inference with data
propagation, standard-domain, sparse/function/subgraph, and Conv-bias-length
checks. The independent runtime trace is decisive where static inference fails
to expose deliberate shape cloaks. “Retain incumbent” is not a promotion and
does not create a new file.

## Cheaper-history and rebuild audit

- **task099:** four distinct cost-397 rank cuts score respectively 0, 38, 0,
  and 0 correct out of 265 in both ORT modes. No giant-free strict-cheaper
  candidate exists in the 29-model historical scan.
- **task279:** the archived nominal cost-357/358 models reprofile at real costs
  4092, 4602, 4092, 4348, and 4603. Two reproduce all 266 known cases only
  with disabled optimization, but every model fails default loading and/or the
  truthful-shape gate. The generator-rule morphology implementation therefore
  does not approach cost 397 without cloaking.
- **task345:** the only cost-365 history scores 153/264 in both modes. A legal,
  truthful, nonnegative-pad control scores 264/264 in both modes but ties cost
  389; it proves legality can be restored, not that cost can be reduced.
- **task239:** cost-374/374/379 variants respectively fail ORT TopK loading,
  score 24/267, and score 2/267. The new zero-blank probe removes the height
  feature, but scores 0/267 and does not reduce actual cost: the feature is
  needed to distinguish in-output background from space outside the output.
- **task075:** a 28-model byte-distinct scan contains no model below cost 345.
  Proportional Slice constants cannot be shared without a larger charged
  runtime tensor. The incumbent itself is cloaked, so it cannot seed a safe
  micro-rewrite.
- **task392:** all five cost-341 token-pruning variants score 0/266 in both
  modes and retain the prohibited lookup architecture. The earlier transparent
  generator-template rebuild costs 9071 and is itself a finite template bank;
  it is neither cheaper nor eligible under the no-lookup requirement.
- **task387:** 57 local models have no lower-bound alternative below current
  cost 337. Folding the fixed `Size(input)` appears to give an arithmetic
  cost-330 graph, but full checker, strict inference, and both ORT modes reject
  it because a one-element shape conflicts with empty `CenterCropPad` axes.
- **task225:** both cost-306 4x4-carrier variants score 0/265. A 26-model scan
  and all 48 available single optimizer passes find no strictly cheaper
  known-correct graph; the fifth carrier row/column is required by the Resize
  sampling paths.

Thus no model reaches the prerequisite intersection of strict cost reduction,
complete known correctness, dual-ORT runtime success, truthful shapes,
standard semantics, and no giant/lookup/UB construction. Candidate fresh
model testing was intentionally not claimed; the fresh 10,000/task evidence
above validates the compiled true rules, not a rejected ONNX.

## Evidence files

- `model_audit.json`: full structural, real-cost, dual-ORT known, runtime-shape,
  lookup/giant/domain/bias audit for exact members and decisive histories.
- `reference_audit.json`: known and two-seed fresh verification of all eight
  independent executable specifications.
- `audit_models.py`, `reference_audit.py`: reproducible audit drivers.
- `build_task239_zero_blank.py`, `task239_zero_blank.onnx`: rejected probe only;
  never integrate it.
- `result.json`: machine-readable final disposition.
