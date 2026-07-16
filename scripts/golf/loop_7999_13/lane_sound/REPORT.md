# SOUND lane A report — baseline 7999.13

## Result

The strict winners are task168 and task344. The conservative cost-based gain is
`+0.7484980265514274`, projecting `7999.13 -> 7999.878498026551`. This number
counts only measured local cost improvements and does **not** assume any
private-zero recovery.

The isolated candidate ZIP is
`scripts/golf/loop_7999_13/lane_sound/submission_7999.13_sound168_344.zip`
(SHA-256 `42318dbcc353e30ebdaf6b5db31a23bd1a585e61ea0bd6c08eeb35c37c37352c`).
It changes only `task168.onnx` and `task344.onnx` relative to the exact
`submission_base_7999.13.zip` baseline.

| task | baseline cost | candidate cost | reduction | local gain | known | fresh generator | decision |
|---:|---:|---:|---:|---:|---:|---:|---|
| 168 | 432 | 416 | 16 | +0.037740327982846 | 265/265 | 5000/5000 | accept |
| 344 | 401 | 197 | 204 | +0.710757698568582 | 266/266 | 5000/5000 | accept |
| 192 | 1609 | 3325 | -1716 | -0.725856806097575 | 265/265 | 5000/5000 | reject |

The acceptance threshold supplied by the user was 95%. This lane deliberately
used the stronger SOUND threshold of 100% on all visible cases and 5000 fresh
generator cases. A single mismatch would have rejected the candidate.

## True rules reconstructed from generators

### task168

Detect every monochrome L triomino: three equal nonzero cells occupying three
corners of a 2x2 box. Starting immediately beyond the missing corner, extend
the same color along the diagonal pointing away from that box until the grid
boundary. Preserve the original grid and the missing corner.

The accepted graph bit-packs rows, detects the four missing-corner
orientations, selects the corresponding arrow, and uses a fixed geometric
diagonal-ray tensor to render the extension. The final 52-input `Einsum` is
large enough to merit scrutiny, but its operands encode the four universal
orientations and boundary rays; they do not contain example grids or expected
outputs. The model has 179 initializer elements, no `TfIdfVectorizer`,
`Hardmax`, example table, custom function, nested graph, or sparse tensor.

The readable reference in `reference_audit.py` matches all 265 visible cases
and 5000 independently generated cases. The ONNX candidate independently
matches the same 265 visible cases and 5000 fresh cases with minimum raw margin
`1643708.0`.

Lower-cost alternatives were rejected instead of relaxing the gate:

- cost 166: 4189/5000 fresh, 811 failures;
- cost 285: 418 fresh failures out of 5000;
- cost 358: one failure in only 100 fresh cases;
- cost 346: 38 fresh failures out of 5000 and four `TfIdfVectorizer` nodes;
- cost 464: 5000/5000, but worse than the cost-432 incumbent.

### task344

Apply one simultaneous orthogonal-neighbor rewrite: every color-2 cell having
an adjacent color-3 becomes zero; every color-3 cell having an adjacent color-2
becomes color 8; all other cells remain unchanged.

The accepted graph is a one-node rank-factor `Einsum` model. The builder
algebraically reuses the same color factor for two roles and preserves the
active generator input columns `{0,2,3,5}` exactly. The float32 factor
reconstruction error is at most `0.0001220703125`, well below the measured
minimum decision margin `313.70709228515625`. Its 197 initializer elements are
local-rule factors, not an output lookup table.

The readable reference matches 266/266 visible and 5000/5000 independently
generated cases. The ONNX graph separately matches 266/266 and 5000/5000.

### task192

Select the most frequent nonzero color A, breaking ties toward the smaller
color. Emit A at a nonzero center cell exactly when A appears in both its
center-inclusive horizontal radius-1 window and its center-inclusive vertical
radius-1 window; otherwise emit zero.

The cost-3325 bitset graph is a generator-SOUND control and passes 265/265 plus
5000/5000. It is rejected because the exact incumbent hash
`e7f9a11b93b611acfa4bba39e90e1ddf24223d50add4277fe9716f21f6ede10c`
is the retained LB-white cost-1609 fallback, not the later documented black
task192 hash. Replacing it would conservatively lose `0.725856806097575`.
The independent lineage review is in
`scripts/golf/scratch_codex_plus10/review_task192/REPORT.md`.

## Structural and runtime gates

For all three audited candidates:

- `onnx.checker.check_model(..., full_check=True)`: pass;
- strict shape inference with data propagation: pass;
- all inferred dimensions static and positive;
- exactly `[1,10,30,30]` input and output;
- standard ONNX domains only;
- zero functions, sparse initializers, nested graphs, banned ops, sequence ops,
  and Conv bias-length issues.

The accepted task168 and task344 models also pass both the repository verifier
and the separate team validator. The latter reports the expected official-like
costs and accepts each against its incumbent. Its only warning is task344's IR
version 7; full ONNX checker, shape inference, onnxruntime inference, all known
cases, and all 5000 fresh cases pass, so this is informational rather than a
runtime failure.

## Archive gates

The candidate archive preserves all 400 baseline members in their original
order, with the original ZIP comment and member metadata. Byte comparison finds
exactly two changed members: task168 and task344. The external archive audit
reports 400 unique ONNX tasks, no duplicate, missing task, or over-limit model,
and `valid: true`. The full-archive Conv bias UB scanner reports zero issues.

Evidence files:

- `winner_manifest.json`: concise adoption manifest and score accounting;
- `reference_audit.json`: readable-rule visible/fresh proof;
- `verify_fresh5000.log`: model visible/fresh/margin proof;
- `structural_audit.json`: checker/static/domain/banned/nested/sequence/bias proof;
- `external_task168.json`, `external_task192.json`, `external_task344.json`:
  independent task validation and cost comparison;
- `zip_build_audit.json`: archive identity, metadata, and changed-member proof;
- `external_zip_audit.json`: independent archive audit;
- `external_zip_compare.json`: independent baseline-vs-candidate ZIP comparison,
  accepting both changed tasks and reproducing `+0.7484980265514274`;
- `full_zip_conv_bias.log`: full-archive bias UB scan.

No root `submission.zip`, score ledger, CSV, `artifacts/`, or `handcrafted/`
file was written by this lane.
