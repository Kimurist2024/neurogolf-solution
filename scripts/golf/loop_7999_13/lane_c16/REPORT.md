# Lane C16 — cost 100–149 strict audit

## Result

No candidate is admissible. C16 contributes **+0.0** and leaves the exact
`submission_base_7999.13.zip` unchanged.

- baseline SHA-256:
  `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`;
- exact bases audited: **7**;
- local/history/mode-reuse probes audited: **31**;
- harvest rows screened: **14**;
- accepted candidates: **0**.

All loadable candidates were remeasured by the actual scorer and checked on
the complete known corpus under both `ORT_DISABLE_ALL` and default ORT. Full
checker, strict shape/data inference, standard domains, banned/nested/function/
sparse checks, Conv-family bias, and runtime intermediate shapes were audited.

| task | exact cost | best apparent lower cost | result |
|---:|---:|---:|---|
| 074 | 135 | 133 | all ten feasible mode-reuse label-sharing variants are 0/267 in both modes |
| 093 | 135 | none | known-complete controls start at 1132 |
| 136 | 135 | none | historical best ties 135 and uses a 58-input Einsum |
| 180 | 139 | 129 | one candidate errors 268/268; the other is wrong 268/268 |
| 221 | 145 | 142 | buffer-shape runtime failure on 267/267 with optimizations disabled |
| 278 | 138 | 135 | buffer-shape runtime failure on 265/265 with optimizations disabled |
| 295 | 137 | nominal 134 | complete-domain factor search leaves 192 sign errors; no model emitted |

## task074 mode-transform result

`Tfeat` and `Bfeat` are exactly related in either direction by a dense 2x2
integer transform, nominally replacing six initializer elements with four.
The exact graph, however, is a single direct-output Einsum with 80 operands and
50 of the 52 legal index letters already active. Each of the eight feature
occurrences requires an independent contraction label. Reusing the only two
free letters couples otherwise independent latent indices and changes the
tensor network.

C16 built both transformation directions with five natural partitions of the
eight uses: all-shared, halves, alternating, adjacent pairs, and cross-pairs.
All ten graphs pass checker and strict inference and measure cost 133, but each
is wrong on all 267 known cases in both ORT modes. Precomputing the transformed
six-element feature in a separate node would add a 24-byte float32 intermediate
to save two parameters, so it cannot improve cost. The truthful D4 and orbit
table controls cost 11622 and 6236 respectively.

## Remaining tasks

- **task093:** exact cost 135 is a memory-free direct tensor network. The
  spec-derived barrier/count renderers are correct but cost 1132–1323.
- **task136:** the alternate history model is known-complete but merely ties
  135 and retains a 58-input Einsum. Other history starts at 1194.
- **task180:** direct output-zero-point compression violates QLinearConv's
  scalar-zero-point contract; the bias alternative cannot distinguish padded
  cells and is wrong everywhere. Removing the zero point is checker-invalid.
- **task221:** the only cheaper measured edit is a shape cloak. It triggers an
  ORT buffer reuse mismatch; correct alternatives cost 151 or 200 and retain
  23–25 false declarations.
- **task278:** cost-135 Min/Max and other shaves depend on incompatible hidden
  allocations and fail with optimizations disabled. Correct conventional
  controls are orders of magnitude larger.
- **task295:** cost 137 is the exhaustive finite-generator-domain solution.
  Searches at nominal costs 117, 125, and 134 fail the complete sign table;
  fp16 initializer substitution is type-invalid for the mixed Einsum.

## Fresh disposition

No strictly cheaper candidate passed complete known correctness and runtime
safety in both ORT modes, so no candidate advanced to independent dual-runtime
fresh 5000. `fresh_evidence.json` records the decisive prior spec/fresh audits.

Root ZIP, CSV, score ledger, and shared models were not modified by C16.
