# task012 sub-650 POLICY90 search — wave417

## Decision

**No sub-650 finalist was admitted.**  The retained task012 candidate remains
the clean 8x8 biased depthwise Conv at cost650, SHA-256
`9aea31a6c01f7af21d893f6e5dde16dc947cdb17088686654f3f568845fbb947`.
It is already independently admitted at known `252/265 = 95.0943%` and fresh
minimum `94.55%` under the four required ORT configurations.

The generator's finite default support is exactly
`7 columns * 7 columns * 4 gravities = 196` states.  POLICY90 therefore needs
at least177 exact states.  The strongest retrained biased kernels below cost650
were already solved to MILP optimality at only `176/196 = 89.7959%`:

| kernel | cost | proven finite optimum | disposition |
|---|---:|---:|---|
| 7x7 | 500 | 176/196 | reject |
| 7x8 / 8x7 | 570 | 176/196 | reject |
| 7x9 / 9x7 | 640 | 176/196 | reject |

The prior cost570 runtime audit also measured fresh `4490/5000 = 89.80%` and
`4464/5000 = 89.28%`, corroborating the finite-support rejection.  Because the
exhaustive finite support is already below90%, no rejected wave417 model was
advanced to a fresh admission audit.

## Exactly three internal workers

1. **Biased crop worker:** generated and ran all399 nonempty contiguous
   rectangular crops of the admitted8x8 weights.  The best crop was8x7,
   cost570, at only `136/196 = 69.3878%`; errors were0.  This is a direct crop
   screen, separate from the stronger retrained MILP boundary above.
2. **No-bias worker:** checked centered/adjacent 7x7, 7x8, 8x7, 7x9, 9x7,
   and8x8 layouts.  For every layout, all196 individual states were
   homogeneous-linearly infeasible for the shared nonzero-color kernel.  The
   literal8x8 bias-drop costs640 and ran `0/196` in all four ORT
   configurations, with zero errors/nonfinite/shape mismatches.
3. **Alternative terminal worker:** proved that a parameter-free rectangular
   pooling terminal cannot emit the center color's nine-cell X.  The target's
   row and column projections each have five coordinates, so any Cartesian
   pooling support containing it has at least25 cells, not9.  Four runtime
   witnesses (identity, 3x3 box, dilated3x3, 1x5 line) all scored0/196 in all
   four configurations.  Multi-node bias synthesis, quantized terminals, and
   shared-kernel reshape routes each require at least one full-grid one-byte
   intermediate (9000 bytes), already above the entire cost target.

All saved ONNX files in this lane are explicitly diagnostic/rejected.  There
is no candidate checkpoint and no projected gain.

## Safety and immutability

- Standard ONNX only; no lookup, fixture correction, private-zero route,
  sparse initializer, external data, shape cloak, or UB was used.
- Canonical static `[1,10,30,30]` input/output was preserved.
- `submission.zip` stayed SHA-256
  `1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231`.
- `all_scores.csv` stayed SHA-256
  `3f9914a0db88302f9e0424d604f9c0e300dc75115570625d296e21b7fcfaf731`.
- `others/` was not touched.

Machine-readable results are in `evidence.json` and the three
`worker*_*.json` files.  `run.py` enforces exactly three subprocess workers.

