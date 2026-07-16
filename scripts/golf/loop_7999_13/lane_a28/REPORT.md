# A28 — task250 / task374 strict optimization

## Outcome

No candidate is eligible for adoption. Verified gain is **0.0**. The root ZIP,
score ledgers, CSV files, and shared handcrafted artifacts were not modified.

The source was Wave 12, whose two members are unchanged from the exact 7999.13
authority:

| task | SHA-256 | memory | params | cost | result |
|---:|---|---:|---:|---:|---|
| 250 | `bd479d52a359a4a1162387298b3f78691e7fa882a771afd5dda7313b92db6b0e` | 279 | 189 | 468 | no safe smaller graph |
| 374 | `93fb94260388ab83bc35043c0ee11ae08b1bf3e8fa962a3b47b08ba73794d24a` | 451 | 30 | 481 | no truthful smaller graph |

## task250

The generator rule was re-read: a 2x2 red box has top-left in `[2,6]^2`; each
gray pixel is moved to the clamped 4x4 ring coordinate. The incumbent compiles
this rule with two coordinate decoders, packed gray row/column codes, two
arbitrary 4x4 feature tables, and a final 2x2 QLinearConv.

The only new strict-cost lead replaces the final QLinearConv by ConvInteger and
removes `yscale`, giving **468 -> 467**. It is structurally clean (full checker,
strict inference, truthful runtime shapes, no errors), but semantically wrong:
both disabled and default ORT are **0/265** known-correct. The missing
requantization threshold turns accumulator values 21--392 into false positive
cells. It is rejected before fresh validation.

The two prior lower static leads were independently re-audited:

- cost 452: float16 `Resize` ROI triggers a type mismatch on **265/265** cases
  under both ORT modes;
- cost 466: disabled ORT triggers **265/265** runtime errors and tracing finds
  two false declarations, although optimized/default ORT happens to run.

The incumbent itself is 265/265 with zero errors in both ORT modes, has zero
declared/runtime shape mismatches, passes full checking and strict
shape/data-propagation inference, and contains no dead/duplicate/proportional/
slice-reusable initializer found by the global exact scans. The 16-element
absent/present tables and their difference each have matrix rank four, so the
previous factor route remains unavailable.

## task374

The generator contains three non-overlapping gray line segments of distinct
length, recolored shortest/middle/longest. The incumbent is known-correct
(267/267, errors 0 in each ORT mode), but it is not a candidate template for
new work under the strict safety rules: runtime tracing finds **nine** false
shape declarations, including graph output declared `[1,1,1,1]` but actually
`[1,10,30,30]`.

The historical one-parameter rewrite (`CastLike` to `Cast`) is algebraically
correct, but it exposes the true 10x10 carrier. Its actual measured cost is
**876**, not the static-looking 480, and it retains the same nine runtime shape
mismatches. Thus it is both costlier than 481 and ineligible under the no-shape-
cloak gate. Earlier exhaustive history contains 26 byte-distinct task374
models and no actually cheaper known-correct member.

## Evidence

- `evidence/model_audit.json`: both ORT known results, actual costs, strict
  checker/inference, runtime shape traces, banned-op/domain/lookup audit.
- `evidence/task250_conv_integer_known.json`: independent team-validator result.
- `evidence/exact_initializer_reuse.json`, `proportional_reuse.json`,
  `slice_reuse.json`, `tensor_mode_reuse.json`: exact initializer searches.
- `winner_manifest.json`: empty accepted set and explicit rejection reasons.

No candidate reached the prerequisite set (strictly cheaper, correct, both-ORT
error-free, truthful). Therefore fresh5000 was intentionally not spent on a
terminally rejected graph.
