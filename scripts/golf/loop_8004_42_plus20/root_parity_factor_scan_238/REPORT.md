# Root parity / exact-factor census (LB 8009.46)

## Outcome

No new strict-lower candidate was found.

The authority archive was `submission_base_8009.46.zip` (SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`).
All 400 archive members were enumerated. Task 310 was deliberately excluded from
candidate discovery because its exact parity factorization is already staged as
SHA-256 `6ccf625a0dca41d5c9cb39ddb41c3756313f2a01ac95f38d70c880c677ccf858`.
The task-310 authority member SHA is
`4eed21efedf2b44e11d2bb748d383275d193144c3c0f8f9f55265c8639e6fdec`.

## Census

- Models enumerated: 400 (399 candidate-eligible, task 310 excluded)
- Initializers consumed by at least one `Einsum`: 858
- Exact small tensors screened: 161 (8--256 elements, rank at least 2,
  integer/bool/float values that are finite exact integers with absolute value at
  most 16)
- Exact-factorizable initializers: 11
- Exact decompositions found: 29
  - full rank-one: 8
  - rank-one across an axis bipartition: 19
  - Walsh support of at most four characters: 2
  - all-equality truth tensors: 0
- Initializers with positive shared-parameter saving: 0
- Strict official-cost-lower candidates: 0

Every one of the 161 screened initializers, including its task, authority SHA,
name, shape, dtype, use count, `Einsum` labels, unique values, classification,
and any exact proof, is recorded in `scan.json`.

## Closest exact decompositions

The table reports the best shared parameter delta for each of the 11 detected
initializers. Factors repeated across tensor axes, graph occurrences, or matching
existing initializers are charged only once.

| task | initializer | shape | uses | best exact class | old params | new params | saving |
|---:|---|---|---:|---|---:|---:|---:|
| 069 | `presence` | 9x1 | 2 | partition rank-one | 9 | 10 | -1 |
| 074 | `T` | 2x2x2 | 18 | four-term Walsh | 8 | 20 | -12 |
| 131 | `red_channel_mask` | 1x10 | 4 | partition rank-one | 10 | 11 | -1 |
| 134 | `fg_mask_h` | 10x1x1 | 1 | partition rank-one | 10 | 11 | -1 |
| 247 | `fg_mask` | 1x10 | 1 | partition rank-one | 10 | 11 | -1 |
| 250 | `gray_sel` | 10x1x1x1 | 3 | partition rank-one | 10 | 11 | -1 |
| 280 | `Bemb` | 1x1x2x10 | 110 | partition rank-one | 20 | 20 | 0 |
| 328 | `Rel` | 2x2x2 | 2 | four-term Walsh | 8 | 20 | -12 |
| 379 | `sel2` | 1x10 | 1 | partition rank-one | 10 | 11 | -1 |
| 382 | `redsel` | 1x10 | 2 | partition rank-one | 10 | 11 | -1 |
| 396 | `os2` | 10x1 | 2 | partition rank-one | 10 | 11 | -1 |

Task 280 is the nearest miss: even after reuse of an existing identical factor,
its factor parameter count is equal to the original. Rewriting it therefore
cannot strictly reduce the official cost; its 110 occurrences do not multiply
initializer cost because the shared factors are stored once.

## Method and gates

`scan_parity_factors.py` reads models directly from the immutable authority zip.
It recognizes only exact reconstructions:

1. full tensor rank-one and every non-duplicate axis bipartition, verified by
   exact array equality;
2. binary-axis Walsh transforms with one to four nonzero coefficients, followed
   by exact reconstruction (covering parity/sign truth tables in this band);
3. all-equality truth tensors, reconstructed as a shared latent-index CP sum.

For each decomposition, the scanner deduplicates bit-identical factors both
inside the decomposition and against other initializers already in the model.
A model is built only if this proves a positive parameter saving. A built model
would then have to pass the full ONNX checker, strict shape inference with data
propagation, and a strict official `memory + parameters` cost decrease before it
is written under `candidates/`.

No decomposition crossed the first (positive parameter saving) gate, so no model
was built and runtime raw pass-through testing was inapplicable. This is safe for
the official-cost gate: the rewrite leaves the `Einsum` output shape unchanged,
so an initializer replacement with zero or negative parameter saving cannot
produce a strict `memory + parameters` reduction.

## Reproduction

```bash
.venv/bin/python scripts/golf/loop_8004_42_plus20/root_parity_factor_scan_238/scan_parity_factors.py
```

Generated census: `scan.json`. The `candidates/` directory is intentionally
empty. Root authority and staging artifacts were not modified.
