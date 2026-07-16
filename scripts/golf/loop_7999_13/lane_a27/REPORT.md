# A27 strict report: task354 and task368

## Outcome

No candidate is admissible. The accepted set is empty and projected gain is
`+0.0`. No shared submission ZIP, score CSV, best-score ledger, or handcrafted
artifact was modified.

The exact Wave12 members are:

| task | SHA-256 | measured cost | decision |
|---:|---|---:|---|
| 354 | `c86ec60a3cf1241903cd6ebdf11f210f24f9e57216927c5654985d8f2d28efe4` | 537 | baseline is shape-cloaked; no safe cheaper replacement |
| 368 | `0d950f5053aa62e7a3208be01514ad061b85580875e0e93aa7ee941cbacaa811` | 521 | baseline is shape-cloaked; no safe cheaper replacement |

## Generator truth

- task354 (`ddf7fa4f`): the top row contains three colored lights. Each gray
  rectangle is aligned to exactly one light and must be recolored with that
  light's color.
- task368 (`e76a88a6`): one sprite is colored with two arbitrary non-gray
  colors; every other sprite is an equally sized gray copy. The colored
  prototype must be copied into every gray footprint.

These rules were taken from the generators, not inferred only from examples.

## Terminal structural gate

Both graphs pass ordinary ONNX full checking and strict shape inference only
because their supplied metadata understates opaque operator outputs:

- task354 declares `gn_f` and `data_clear_f` as `[1,1,1,1]`, although
  `GroupNormalization` preserves the fixed `[1,10,30,30]` input shape. It also
  declares the graph output `[1,1,1,1]`, while runtime returns the full padded
  grid `[1,10,30,30]`.
- task368 declares `gn` as `[1,1,1,1]`, although that
  `GroupNormalization(input, ...)` runtime tensor is `[1,10,30,30]`.

Therefore neither exact member is a permissible template under the no-shape-
cloak gate. Retaining one honest float32 GroupNormalization output would expose
36,000 scored bytes, already far above these declared costs; task354 contains
two such outputs. This does not prove a universal lower bound for a completely
new graph, but it rules out metadata-only and local algebraic reductions of the
incumbents.

## Search and algebra audit

- Every node in both exact bases is output-reachable; neither contains a
  duplicate initializer.
- The only syntactically removable task354 node is the `Identity` hiding the
  `CenterCropPad` target. Feeding its unchanged initializer directly makes the
  real dimension 12 inferable and fails full checking due to the existing fake
  dimension 1. The isolated probe is
  `candidates/task354_no_identity.onnx`; it is rejected before runtime testing.
- The archived task354 static-floor-531 graph
  (`65ed30c9...`) actually measures 560, above 537, and retains the same cloak.
- Five historical task354 rows bottom out at actual cost 560 (the next static
  floors are 540, 555, 863, and 2064).
- Four historical task368 rows bottom out at static floor 522, one point above
  521. No dead node, duplicate constant, or one-parameter reuse exists in the
  exact 521 graph. Its dynamic two-channel kernel feature is semantically
  required to preserve the arbitrary two-color prototype.

The exact task368 baseline already has complete independent evidence at
265/265 known and fresh 5000/5000 with zero runtime errors under each of default
ORT and `ORT_DISABLE_ALL` (same SHA, C9 evidence). That correctness does not
override the structural rejection. task354 candidates were not sent to an
expensive fresh gate after the cheaper-looking probes failed cost/checker and
the shape-cloak gate; no rejected probe is being proposed under the >=95%
exception.

## Evidence

- `evidence/audit.json`: exact hashes, measured declared costs, full/strict
  checks, reachability, duplicate-initializer scan, explicit cloak witnesses,
  and historical floors.
- `winner_manifest.json`: empty accepted set.
- `build_candidates.py` and `audit_lane.py`: reproducible isolated build/audit.

Final decision: **NO_WINNER**, gain `+0.0`.
