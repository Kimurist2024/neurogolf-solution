# Target mid11 audit — tasks 156 / 284 / 363 / 368

## Outcome

No candidate is admissible. The accepted set is empty and the projected gain
is **+0.0**. No root ZIP, score CSV, best-score pointer, or
`artifacts/handcrafted` file was modified.

The exact `submission_base_8004.50.zip` members are:

| task | SHA-256 | cost | known, both ORT modes | private-zero catalog | decision |
|---:|---|---:|---:|---|---|
| 156 | `b5457e5da157641c13690ea0a3a0fc2664aeccf5f49e7c1bc233e11f5121c260` | 556 | 265/265 | outside | clean practical floor; no cheaper history |
| 284 | `d8f60072c5f0f2ab7730a1bf26ce2ed68ca0645b5e1a909c489963817738c7b0` | 518 | 266/266 | outside | shape cloak; no cheaper actual-cost history |
| 363 | `aec5b5333bb9fae7c7718096c7640bc6b6ae61c3ca82d0cef8e5b9c70fd95607` | 513 | 265/265 | outside | shape cloak and non-identifiable fixtures; no cheaper history |
| 368 | `0d950f5053aa62e7a3208be01514ad061b85580875e0e93aa7ee941cbacaa811` | 521 | 265/265 | outside | shape cloak; no cheaper history |

All four exact members pass full ONNX checking, strict shape/data propagation,
standard-domain, banned-op, nested-graph/function/sparse, and Conv-family bias
UB checks. These generic checks do not override the explicit runtime-shape
traces below.

## Generator rules

- task156: find the two solid yellow rectangles, preserve their borders, and
  fill the smaller interior with color 1 and the larger interior with color 2.
  The result is invariant to the optional vertical flip. The decoded reference
  passed 5,000/5,000 fresh generator cases.
- task284: two distinct-colored collinear seeds grow inward, end in
  perpendicular five-cell caps, and add two serifs; the whole result may be
  transposed. The spec reference passed 266/266 known, 3,000/3,000 fresh, and
  an exhaustive 1,960-case boundary-parameter sweep.
- task363: restore the red exemplar to black, find every legal translation of
  the diagonally-connected sprite in the black/gray 10x10 canvas, and paint
  those occurrences red. The pure rule passes 5,000/5,000 legal fresh cases.
- task368: extract the unique two-color 3x3, 3x4, or 4x3 prototype and copy its
  exact pattern into every equally-sized gray footprint. The exact incumbent
  passed 5,000/5,000 fresh cases under each ORT mode with zero errors.

## Runtime-shape truthfulness

The all-node-output tracer found:

| task | declared cost | runtime/declaration mismatches | observed intermediate bytes | parameters | truthful observed cost |
|---:|---:|---:|---:|---:|---:|
| 156 | 556 | 0 | 330 | 226 | 556 |
| 284 | 518 | 7 | 57,016 | 53 | 57,069 |
| 363 | 513 | 7 | 90,915 | 83 | 90,998 |
| 368 | 521 | 2 | 45,476 | 40 | 45,516 |

task284 hides the full input/output and 56-write ScatterND index/update
tensors behind `CenterCropPad` declarations. task363 hides full
`GroupNormalization`, data, index, and output tensors. task368 declares its
`GroupNormalization` and following cast as 1x1x1x1 while runtime produces
1x10x30x30. Therefore none of those three incumbents is a permissible base for
a metadata or local-algebra shave under the truthful-runtime-shape gate.

## History gate

The loose/local history was SHA-deduplicated across `scripts/golf`, `others`,
and `artifacts`, then reconciled with the retained archive sweeps:

| task | unique SHA | incumbent | lowest non-incumbent actual cost | strictly cheaper |
|---:|---:|---:|---:|---:|
| 156 | 38 | 556 | 556 | 0 |
| 284 | 63 | 518 | 521 | 0 |
| 363 | 61 | 513 | 513 tie (next 514) | 0 |
| 368 | 72 | 521 | 522 | 0 |

For task284, 23 lower-static candidates were already independently profiled;
their real costs range from 521 upward. The lowest static declarations depend
on the same shape-cloak family. For task156, the incumbent has no unused or
duplicate initializer and its 180-parameter final 2-to-10 3x3 quantized kernel
cannot be factored without adding at least a much larger spatial intermediate.
For task368, all nodes are output-reachable, there is no duplicate initializer,
and the arbitrary two-color prototype requires the dynamic two-channel kernel
feature. The nearest archive variants all cost 522.

## task363 specification conflict

The legal random generator relation is not identifiable from the input for two
fixed stored fixtures. A documented witness adds a valid extra translation to
the second fixture: it yields the exact same input but a different legal
output. Consequently, the pure generator solver is 5,000/5,000 on fresh legal
instances but only 263/265 on stored cases. The exact cost-513 compatibility
model passes 265/265 stored cases and still reaches 4,979/5,000 fresh in both
ORT modes, but it also has seven shape mismatches. No deterministic input-only
model can simultaneously satisfy both relations without a fixture-specific
shim.

## Decision

There is no model that is strictly cheaper than the exact 8004.50 member while
also satisfying known 100%, dual-ORT runtime zero, truthful static/runtime
shapes, strict inference, UB0, no lookup/cloak/giant contraction, and the
two-seed fresh threshold. No candidate was copied into a promotion ZIP.

Machine-readable evidence is in `result.json`; exact baseline members are in
`baseline/`.
