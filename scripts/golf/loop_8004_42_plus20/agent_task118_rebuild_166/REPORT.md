# task118 true-rule rebuild lane 166

## Outcome

No candidate is safely adoptable. The winner set is empty and projected gain
is `+0.0`. Root `submission.zip`, `all_scores.csv`, `others/`, and docs were
not changed.

The immutable authority is LB **8009.46**:

- ZIP SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- task118 member SHA-256:
  `a7c763ba3468863d1cebdf97522fc613052ab5d435af51b8d9035d413c096ab8`
- task118 actual cost: `3031 memory + 634 params = 3665`

## Decoded rule

The Sakana reference was expanded and checked against all train, test, and
arc-gen examples: **267/267** exact grid matches.

The generator first creates random black/gray static and then attempts four
non-overlapping plus-shaped crosses with one common radius `L in {2,3}`. A
cross cell over black becomes red2; a cross cell over gray becomes cyan8. The
input erases cyan8 back to gray5.

The deterministic reference therefore:

1. Enumerates radius-2 plus supports, then radius-3 supports, whose cells are
   red2/gray5.
2. Finds a disjoint set of supports whose union covers every visible red2
   cell.
3. Changes gray5 to cyan8 on those supports and leaves everything else intact.

This is a bounded global exact-cover inverse, not a purely local transform.
The reference reaches `4876/5000` and `4846/5000` on two new generator seeds.

## Formal non-injectivity

No deterministic all-input-equivalent ONNX exists for the generator relation.
Consider a 10x10 all-gray static grid, radius 2, with all four attempted centers
equal. The first center is accepted and the remaining attempts conflict. If
the repeated center is `(3,3)` versus `(6,6)`:

- both generated inputs are the same all-gray grid because every cyan cross
  cell is erased to gray;
- the generated outputs contain cyan pluses at different centers and are
  different;
- both latent executions have positive generator probability.

The machine-readable witness records one input hash and two distinct output
hashes in `result.json`. Thus fresh accuracy can exceed 90%, but no fresh sample
can close a private-zero guarantee over the hidden distribution.

## ONNX controls

Two specification-derived controls were independently audited. Neither uses a
lookup table, giant Einsum, custom domain, nested graph, shape cloak, or
Conv-family bias mismatch.

| model | actual memory + params = cost | node-shape mismatches | known, each of 4 ORT modes | fresh5000, each mode | decision |
|---|---:|---:|---:|---:|---|
| authority | 3031 + 634 = **3665** | **37** | 267/267 | 4319/5000 (86.38%) | fixed incumbent only |
| observable rule | 9009 + 133 = **9142** | 0 | 267/267 | 4814/5000 (96.28%) | reject cost/private guarantee |
| full-ROI control | 51068 + 282 = **51350** | 0 | 267/267 | 4833/5000 (96.66%) | reject cost/private guarantee |

The four runtime modes are ORT default, `ORT_DISABLE_ALL`, basic/minimal, and
extended. For every model, all known and fresh raw tensors are byte-equal to
the disabled-mode result across all four modes. Runtime errors, nonfinite
outputs, and values in the unsafe `(0,0.25)` margin are all zero.

Both independent actual-cost profilers agree exactly on all three costs. Full
checker, strict data propagation, static positive shapes, standard domains,
and Conv UB0 pass for both new controls. Direct all-node traces report zero
runtime/declaration mismatches for both controls.

The authority's 37 runtime/declaration shape contradictions explain why its
3665 cost is far below a normal reconstruction. That incumbent defect is
evidence, not permission to admit another shape-cloaked descendant.

## Structural floor

The smallest truthful observable-rule architecture first decodes the
generator's maximum 25x28 region:

- f32 color-code plane: `25 * 28 * 4 = 2800` bytes;
- required uint8 cast: `25 * 28 = 700` bytes;
- subtotal before any center detector: **3500 bytes**.

The authority's entire budget is 3665. After the observable-rule model's 133
parameters, only 32 cost units remain, which cannot hold a radius-2/3 center
score, length selector, NMS/exact-cover state, and paint mask. The measured
complete truthful form costs 9142. Direct float Conv scoring is still larger,
and exact posterior set-cover requires up to 504 center positions and
combinatorial hypotheses.

## Decision

The observable rule clears the relaxed fresh90 threshold, but it is +5477 cost
and cannot receive an all-hidden/private-zero guarantee because the generator
is non-injective. The full-ROI control is even more expensive. The only
sub-3665 family is the incumbent's shape-cloaked/cropped selector lineage,
which has poorer fresh accuracy and cannot be treated as a sound rebuild.

No ZIP was built and no candidate was promoted.

## Evidence

- `audit_lane.py`: reproducible rule, collision, structural, cost, four-mode,
  known, margin, and fresh audit.
- `result.json`: complete machine-readable results.
- `winner_manifest.json`: empty safe promotion manifest.

