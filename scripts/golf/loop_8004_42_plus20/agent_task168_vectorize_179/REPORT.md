# task168 vectorize / initializer absorption audit

## Outcome

**NO STRICT-LOWER CANDIDATE (pre-gate).** The 8009.46 authority remains the
winner at `memory=237`, `params=178`, `cost=415`, payload SHA-256
`642cba5c350b35774bf98e427ca858a675cd8dd483ea6d1b2ec7e13287739b92`.

No model is proposed for merge. Root `submission.zip` and `all_scores.csv` were
not changed.

## Authority reproduction

- ONNX full checker: PASS
- strict shape inference with data propagation: PASS
- truthful I/O: float `[1,10,30,30]` to float `[1,10,30,30]`; no stale declared value-info
- official local scorer: correct, `237 + 178 = 415`
- known corpus: all 265 (`train=2`, `test=1`, `arc-gen=262`)
- Conv-family bias UB: 0 (the sole Conv has no bias input)
- nodes / initializers: 54 / 16
- functions / sparse initializers: 0 / 0

The reproducible result and every intermediate tensor's scored byte count are
in `audit/result.json`. Re-run with:

```bash
.venv/bin/python scripts/golf/loop_8004_42_plus20/agent_task168_vectorize_179/audit_task168.py
```

## Exact vectorization lower bound

The shared prefix costs 103 bytes through `seed0..seed3`. Downstream branch
outputs cost `34,33,34,33` bytes, respectively (134 total).

Vectorizing `ReduceMax/ArgMax/Cast/Log/Selu/Concat` changes shapes but not the
number or dtype of output elements, so the downstream 134-byte sum is
unchanged. Starting from the four existing `[7]` seeds, a four-way vectorized
path must first materialize a `[4,7]` uint8 tensor (+28 bytes). Pairing the
branches with identical shift/Selu attributes still needs two `[2,7]` tensors
(+14 each). Therefore both direct vectorization forms have cost at least 443.

The differing operations are also semantically required:

- only branches 0 and 2 shift the reduced maximum;
- Selu gamma signs differ between `(0,2)` and `(1,3)`;
- row transforms are `6-r` for `(0,1)` and `r-8` for `(2,3)`.

Selectors/splits for those differences can only raise the bound.

## Initializer and algebraic audit

### Rank-1 `Xrow`

`Xrow[1,10,1,8]` has multilinear rank one. Factoring its 80 parameters into
10+8 saves 62 parameters, but Conv then consumes a reconstructed float32
80-element intermediate, which costs 320 memory. Net delta is at least +258
(cost at least 673).

### `Bc` and `C1T`

`Bc[30,2]` has rank 2. A factorization inside the final Einsum needs at least
`30*2 + 2*2 = 64` parameters, versus the dense 60 (+4).

`C1T[3,2,2]` has mode ranks `[2,2,2]`. Even an optimistic CP-rank-2 encoding
needs `3*2 + 2*2 + 2*2 = 14` parameters, versus the dense 12 (+2).

The descriptor axis initially appears compressible because

```text
M(a,b,r) = [[5a - 10b - 11r, -5a], [-5.5a, 0]]
```

depends only on `a` and `q0=10b+11r`. This does **not** yield an exact shared
2-vector: the second occurrence is multiplied by `sd=[0.9,1,0.9]`, so it
depends on `a` and `q1=10b+9.9r`. The coefficient map from `(a,b,r)` to
`(a,q0,q1)` has rank 3. Thus the current shared three-component descriptor is
minimal under the scoped raw-equivalent contraction. A length-2 control also
leaves `sd` length 3 and fails the Einsum dimension at runtime.

### Scale/sign tensors and coordinate rebase

- Absorbing `sd(3)` and `sc(2)` into a second `C1T(12)` costs +7 parameters.
- Slicing `sn2`/`sc` from another initializer saves at most 2 parameters but
  creates at least one 8-byte float intermediate.
- Rebasing the affine coordinate in `Bc` cannot remove `six_i8/eight_i8`:
  `Bc` is shared by row and column roles, and the required shear does not
  commute with the `sn2` and `sc` gauges.

## Gate decision

There were zero strict-lower transformations after the algebraic/cost
pre-gate. Consequently there is no candidate on which to claim raw
equivalence, known correctness, or fresh correctness. Per the assignment's
explicit pre-gate exit, fresh seeds 0..3 × 10,000 were not run. No private-zero,
lookup, sparse, shape-cloak, or policy-relaxed model was emitted.

## Root guards after the audit

- `submission.zip`: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `submission_base_8009.46.zip`: same hash
- `all_scores.csv`: `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`

