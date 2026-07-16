# A33 task132/task222 exact-gauge report

## Outcome

One strict winner was found against immutable Wave17 archive SHA-256
`74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534`.
Only lane-local artifacts were written; the aggregate archive was not changed.

| task | baseline | candidate | gain | decision |
|---:|---:|---:|---:|---|
| 132 | 316 | 312 | +0.0127390258 | accept |
| 222 | 380 | - | 0 | stop: incumbent itself is unsound |

## task132 exact gauge and initializer reuse

The existing one-node network uses a binary comparison transition tensor
`Q[f,C,q,v]`, endpoint factors `PC`/`L`, and the color bilinear matrix `A`.
The candidate applies invertible state gauges

```text
F = [[0.5, 0], [-1, 1]]
C = [[1, 0], [2, 2]]
Q' = F Q C^T
PC' = F^-T PC
L' = C^-T L
```

so every spatial comparison contraction is unchanged.  The same transformed
tensor has the exact repeated-index view

```text
Q'[m,t,m,t] = [[0.5, 1], [-1, 0]] = A / 1e10.
```

Scaling the shared `H` initializer by `1e5` therefore lets both `A` operands
reuse `Q'`; `A` is deleted.  This removes four charged parameters.  There are
still one node, 47 Einsum inputs, and the same 50 index letters.  No operand,
new index, node output, lookup table, sparse initializer, or shape declaration
was added.

## Gate evidence

- Cost: memory 0, params/cost 312; official-like visible correctness true.
- Known: 267/267 under `ORT_DISABLE_ALL` and 267/267 under default ORT, zero
  errors in both.
- Fresh generator: 5000/5000 under each ORT mode, zero generation/runtime/output
  errors.
- Margin: stable, minimum positive 0.8670001029968262.
- Structure: ONNX full check and strict shape inference with data propagation
  pass; runtime declaration mismatches 0; no banned/nonstandard/nested/function/
  sparse constructs; no Conv-family nodes.
- External validator: known 267/267, cost 312, no warnings, verdict
  `ACCEPT_STRICT`.  Its arbitrary out-of-generator random differential has one
  threshold difference in 500; this is explicitly allowed because the strict
  generator-domain dual-ORT run is 5000/5000.
- Lane ZIP SHA-256
  `42db05f21fb3a3768a9491ccc5084601066b17af50e110f9015610416f0ccadb` is a
  valid 400-member archive with only `task132.onnx` changed and Conv UB count 0.

## task222 stop

The true rule selects a solid 9..16-cell same-color rectangle from dense 16x16
noise.  Wave17's zero-memory rank-8 analog matcher is only 2813/3000 fresh.
Every previously deleted rank component is known-wrong, and removing the
output projection fails all known/fresh cases because it loses background
recoding.  A strict sound rebuild below cost 380 is not available, so no
task222 artifact was promoted.
