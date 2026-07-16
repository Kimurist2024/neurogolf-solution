# task354 targeted reduction — 8012.15 authority

The combined candidate reduces task354 from **497 to 461** and gains
`ln(497/461) = +0.075191983100`.

## Reduction

- Reuse the already-routed column-4 and column-7 colour choices at the two
  neighbouring boundary columns. This removes two redundant `And` tensors.
- Transpose the 10-element top row before three-band broadcasting, instead of
  transposing the resulting 30-element band tensor.

These two changes save 36 units of intermediate memory without changing the
parameter count.

## Audit

- known: 266/266 in all four ORT configurations
- fresh: 2 seeds × 2,000 cases × 4 ORT configurations, all correct
- minimum accuracy: 100%
- runtime errors, nonfinite values, output-shape failures, small positives,
  cross-configuration differences: all zero
- checker/full-check, strict shape inference, finite initializers, banned-op
  scan: pass

Candidate:
`candidates/task354_combined.onnx`

SHA-256:
`c45a5760c95e9bd22268f927e2f98cda8195c5d87e9dcca533977574b58b3a75`

Evidence:
`search_evidence.json`, `audit_evidence.json`

## Risk classification

task354 is not one of the newly confirmed black candidates and its authority
lineage has prior LB-white probes. However, the authority already uses a legacy
declared/runtime shape mismatch; this candidate preserves that exact lineage.
It is therefore admitted under the user's nonblack POLICY90 rule, but remains
separate from truthful/no-cloak safe candidates and should be tested with an
individual probe ZIP before a cumulative merge.

No root submission, score CSV, or `others/` checkpoint was modified by this
lane.
