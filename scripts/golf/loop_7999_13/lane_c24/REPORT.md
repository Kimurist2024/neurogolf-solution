# Lane C24 report: task363 / task388

## Decision

No model was promoted. The accepted score gain is **+0.00**.

Neither task produced a candidate that was simultaneously strictly cheaper,
correct on all known examples in both ORT modes, generator-rule sound,
runtime-shape truthful, and free of runtime/session errors. Accordingly, no
candidate entered the final external-validator gate.

## Exact baseline remeasurement

| task | exact SHA-256 | declared memory + params | declared cost | maximum-size runtime probe | shape mismatches |
|---:|---|---:|---:|---:|---:|
| 363 | `aec5b5333bb9fae7c7718096c7640bc6b6ae61c3ca82d0cef8e5b9c70fd95607` | 430 + 83 | **513** | **90,998** | 7 |
| 388 | `f4450fa21dfce9e893c6b70646d43590ff60fb02b3d6a21856c97438061945b3` | 291 + 20 | **311** | **55,735** | 14 |

Thus the exact 7999.13 cost for task363 is confirmed as 513 under the current
declared-shape scorer. It is not a truthful static-shape cost: its real
intermediate tensors consume 90,915 bytes on the fixed 10x10 domain, before
the 83 parameters are added. The output and six other tensors are declared
smaller than their runtime shapes.

Task388 has the same issue. On its generator-maximum 6x6 input, intermediate
tensors consume 55,715 bytes before parameters, despite the 291-byte declared
memory result. Both exact graphs pass ONNX full checking and strict inference;
those checks do not expose these deliberate declarations, while the all-output
runtime tracer does.

## Generator rules and references

### task363

The generator places a diagonally connected sprite several times over a
black/gray 10x10 canvas. The first occurrence is red in the input. A pure
input-derived solver restores red to black, normalizes the red exemplar to its
bounding-box origin, finds every translation whose required cells are
black/red, and paints every such occurrence red.

That rule is verified on **5,000/5,000 fresh generator cases**, with zero
errors. It matches only **263/265** stored examples: `train[0]` and `train[1]`
are inconsistent with the random generator's legality relation.

This is not merely a missed edge case. For `train[1]`, adding the discovered
anchor `(1,3)` produces the exact same input but a different output, while the
alternate three-location parameterization is complete, disjoint, and legal.
Therefore no deterministic input-only ONNX can satisfy both every legal
generator instance and every stored fixture. A stored-fixture compatibility
branch is necessarily outside the pure true rule.

The exact cost-513 model demonstrates the resulting risk: it passes all 265
known examples in both ORT modes, but fresh testing reaches only
**4,979/5,000** in each mode, with 21 wrong outputs and zero runtime errors.

### task388

For square size `s` in 2..6, each input column containing a nonzero pixel is
painted cyan (8), the original non-cyan foreground pixels override cyan, and
the resulting `s x s` tile is repeated 2x2. The readable reference passes
**266/266 known** plus **5,000/5,000 fresh** cases with zero errors.

The exact cost-311 model also passes 5,000/5,000 fresh in both ORT modes, but
it under-declares 14 runtime tensors and uses `CenterCropPad` to hide dynamic
ROI shapes. It is rejected by the no-shape-cloak requirement.

Reference details and seed policy are in `reference_audit.json`; exact-model
fresh totals and failure seeds are in `fresh_exact_audit.json`.

## Strict-shave and archive results

### task363

The deduplicated retained history contains eight models with static cost
floors 514, 517, 525, 579, 704, 707, 735, and 4284. None is below the exact
513 baseline. The nearest three were re-audited and all retain seven or eight
runtime/declaration shape mismatches.

The generator-spec core costs 680, passes fresh 5,000, but fails the two
inconsistent known fixtures and still has two shape mismatches. The fully
shape-truthful conventional control costs 42,866. Neither can be promoted.

### task388

The prior exact-factor history scan covered 21 SHA-distinct models. Its only
actual-cost candidate below 311 has SHA
`d10cd57a8446a1548dd15aaaf93504f8ccf011d5b530b6fa68f40da20b344590`
and nominal cost 137, but every known execution errors in both ORT modes
(0 right, 266 errors), with 15 shape mismatches.

The five retained static-leading archives remeasure to actual costs 9452,
9310, 311, 312, and 314. The cost-9310 model is wrong on all known examples;
the others are not strictly cheaper and each has 14 or 15 shape mismatches.
The other suspicious archive uses a 24-input giant Einsum and is rejected as
a structural lookup/abuse lineage.

The fully shape-truthful conventional task388 control costs 6,468 and passes
all 266 known examples in both modes. It is 6,157 units above the exact
declared cost, so it does not enter fresh/external promotion testing.

Complete model-level measurements, operator scans, known totals, shape traces,
and hashes are in `model_audit.json`. Archive provenance is summarized in
`history_evidence.json`.

## Gate disposition

There is no strictly cheaper sound candidate. Candidate fresh-5,000 and
external-validator runs were therefore not performed. The 5,000-case runs in
this lane are explicitly reference and exact-baseline audits, not acceptance
tests for a promoted candidate.

No banned operator, nonstandard domain, local function, sparse initializer,
giant Einsum, lookup signature, or unsafe convolution bias was tolerated in a
potential winner. The one giant-Einsum archive was rejected before testing.

## Root integrity

The forbidden artifacts remained unchanged:

- `submission_base_7999.13.zip`: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- `all_scores.csv`: `3f9533f472a2153e12daeea4936aa7be3f47902a8fdb1621c31f778f6d009665`
- `best_score.json`: `551409d40c18ef80a9ae7e89a6a0e567aa2474924018225e29639b32c0627e72`
- `artifacts/handcrafted` aggregate (402 files): `5344ea88ff3e24509ed49fbc51b613ced484c8000513ee45060a6ce0b7ddbf69`

All C24 artifacts are confined to
`scripts/golf/loop_7999_13/lane_c24/`.
