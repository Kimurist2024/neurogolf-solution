# C11 archive quick-lead SOUND report

## Outcome

No safe winner was found for tasks 077, 079, 090, 096, 143, or 153.
The accepted manifest is empty, projected gain is `+0.0`, and no root ZIP,
CSV, score ledger, or shared handcrafted artifact was changed.

The immutable baseline is exactly `submission_base_7999.13.zip`, SHA-256
`a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`.
All six baseline models were extracted directly from that archive. Real cost is
the sum of visible intermediate tensor bytes and parameter elements; input and
output tensors are free.

| task | exact cost | cheapest audited lead | lead cost | apparent saving | decision |
|---:|---:|---|---:|---:|---|
| 077 | 3364 | r07 | 3345 | 19 | shape-cloak and fresh reject |
| 079 | 210 | r02 | 209 | 1 | shape-cloak/lookup and fresh reject |
| 090 | 1050 | r02 | 348 | 702 | lookup/shape-cloak; sound r06/r07 fail fresh |
| 096 | 1275 | r01 | 1111 | 164 | default-ORT, shape-cloak, and dynamic-bias reject |
| 143 | 212 | r02 | 148 | 64 | lookup carrier; catastrophic fresh reject |
| 153 | 237 | r02 | 236 | 1 | short QLinearConv bias UB reject |

Every candidate and baseline was remeasured, not trusted by filename. All
known-example results reported below used fresh sessions under both
`ORT_DISABLE_ALL` and default optimization, unless a default session could not
be constructed. Full ONNX checking, strict shape inference with data
propagation, standard-domain, banned-op, nested-graph/function/sparse, and
Conv-family bias audits were run independently in `candidate_audit.json`.

## Generator rules and archive provenance

- **task077:** infer the hidden yellow rectangles from their exposed red
  fragments after the static obstruction. r07 is the prior static-3345 family.
- **task079:** identify the most frequent of the repeated 3x3 sprite types and
  output that sprite template/color. r02 is the historical
  `lane26/task079_clamped_ratio.onnx` lead.
- **task090:** find the unique maximum-area all-zero axis-aligned rectangle and
  fill it with pink 6. r02/r04 came from the cost-348/400 lookup families;
  r06/r07 came from the cost-1004/1016 algorithmic C3/7904 families.
- **task096:** reconstruct the cross-quadrant pattern indexed by the ordered
  colors and segment lengths. r01/r02 are the historical cost-1111 and
  task096-improved-v2 leads.
- **task143:** read the boxed creature template at top-left, locate the matching
  colored creature, recolor the target with the boxed color, and turn the
  original gray. r02 is the historical cost-148 carrier.
- **task153:** combine the two colored 3x3 partial patterns into the full 3x3
  answer. r02 is the historical current-best-revalidated cost-236 lead.

## Terminal structural rejections

The known sets are not enough to establish the actual rules:

- task077 r07 is 266/266 on both runtimes, but ten declared shapes contradict
  execution (`cp_1` through `cp_9` and `y30`). Its executed intermediates total
  9980 bytes on one known example while only 3254 bytes are charged. The prior
  independent fresh result is only 4807/5000.
- task079 r02 is 266/266 on both runtimes, but five false shapes shrink
  1x10x30x30 carriers and 10-way selectors to scalar-like declarations.
  Executed intermediates total 63137 bytes versus charged memory 144. It also
  uses Hardmax and a 17-input Einsum, and its prior fresh result is 4962/5000.
- task090 r02/r04 are 267/267 on both runtimes but use twelve
  `TfIdfVectorizer` nodes and respectively five/four false tensor shapes; r04
  additionally has a giant Einsum. These are lookup/shape-cloak candidates and
  are terminally ineligible regardless of their apparent savings.
- task096 r01/r02 are 266/266 only with optimization disabled. Default ORT
  rejects both because a `CenterCropPad` shape has one element for two axes.
  They also carry false runtime shapes; the QLinearConv weight is dynamically
  constructed, so its output-channel count and exact bias safety cannot be
  established. r02 additionally uses two `TfIdfVectorizer` nodes.
- task153 r02 is 265/265 on both runtimes and its memory is truthful, but its
  QLinearConv produces 10 channels with a bias initializer of length 9. That
  one-element truncation is the entire apparent saving and is forbidden
  out-of-bounds/undefined behavior; restoring the tenth bias removes the gain.

## Independent fresh-5000 dual-runtime results

Only the algorithmic task090 r06/r07 leads and the shape-truthful task143 r02
survived far enough to merit a new independent fresh audit. Seeds are distinct
from all cited historical runs, and every run records zero execution errors.

| candidate | disable-all | default ORT | errors | decision |
|---|---:|---:|---:|---|
| task090 r06, cost 1004 | 4766/5000 | 4777/5000 | 0 / 0 | reject |
| task090 r07, cost 1016 | 4786/5000 | 4782/5000 | 0 / 0 | reject |
| task143 r02, cost 148 | 2/5000 | 3/5000 | 0 / 0 | reject |

The task090 candidates implement a close heuristic rather than the unique
global maximum rectangle rule. The compact task143 graph is a known-distribution
lookup carrier rather than a creature-matching algorithm.

No graph reaches the required intersection of strictly lower real cost,
complete known correctness, truthful structure, dual-runtime safety, and exact
independent 5000/5000 validation. Evidence is in `candidate_audit.json`,
`fresh_audit.json`, `rejected_manifest.json`, and the empty
`winner_manifest.json`.
