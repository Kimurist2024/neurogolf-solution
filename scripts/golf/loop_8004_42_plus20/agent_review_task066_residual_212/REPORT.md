# task066 residual cost-551 candidate — independent review 212

## Verdict

**PASS / safe exact parent pass-through.**  The candidate removes ten official
parameter units without changing the task066 parent semantics.  I found no
new lookup, runtime, shape, nonfinite-output, or private-support risk.

- Parent: `others/71407/task066.onnx`
- Parent SHA256: `2e3bd402f667062b32858d3a11182d3e8050d833d2974d1d37fbadd688f4648b`
- Candidate: `../agent_task066_residual_208/task066_residual_cost551.onnx`
- Candidate SHA256: `622b3b28271806949bb18e8b9517335d49cb0383410caf36a19e064d95798dd3`
- Official parent profile: memory 346, params 215, **cost 561**
- Official candidate profile: memory 346, params 205, **cost 551**
- Incremental gain: `ln(561/551) = +0.01798609636978192`

This review edited neither the root submission/CSV nor `others/71407`.

## Exact protobuf delta

The deterministic protobuf comparison permits exactly these changes:

- node 22 / `Gv`: replace the `greenhalf10[d]` Einsum operand with a
  contraction of already-live `Uchan`, `Vchan`, `Trow`, `Tcol`, and `z1`;
- node 23 / `Gh`: the same replacement;
- remove only the ten-element `greenhalf10` initializer.

All other 75 nodes, 17 common initializers, graph/model shell, graph I/O,
value-info, opsets, functions, and metadata are protobuf-identical.  No new
initializer or node was added.

## Independent selector proof

Writing the added contraction as three separable factors gives:

```text
A[d] = sum(a,j) Uchan[a,d] Trow[a,j] Tcol[a,j] = Uchan[0,d]
E[d] = sum(e,k) Vchan[e,d] Tcol[e,k] z1[k]       = Vchan[2,d]
F    = sum(f,l,m) Uchan[f,l] Vchan[f,l] Tcol[f,m] z1[m] = -1

A[d] E[d] F = Uchan[0,d] Vchan[2,d] (-1) = e3[d]
```

I expanded every indexed term, rather than accepting the lane's contraction.
There is exactly one nonzero product in the added selector:

```text
(a,j,e,k,f,l,m,d) = (0,0,2,1,2,0,1,3), value = +1
```

Every factor and partial selector value is in `{-1,0,1}`.  Thus no
cancellation or floating rounding is involved in reconstructing color 3.

The offline reconstructed unused entries can be spelled `-0` while the old
initializer stores `+0`; they are numerically zero.  This does not create a
semantic hole:

- the sole nonzero color coefficient is exactly `+1`;
- on the generator domain, `Gv/Gh` are nonnegative integer sums bounded by
  `2*(2**20-1) = 2,097,150 < 2**24`, hence exactly representable in float32;
- a directed ORT audit including all-zero and misaligned-zero inputs observed
  no signed-zero raw difference in any of the four runtime configurations;
- the following `Cast(Gf -> uint32)` would erase either zero sign before all
  bit-mask logic even if an unobserved contraction order produced `-0`.

## Complete support proof remains valid

The previously completed geometry artifact is pinned at SHA256
`653bb75258ef8e80d9967d1b64ac8a61d75aff0400b38bdb524c1fae447c121f`.
It exhausts 15,336 S tuples and 449,928 U tuples, or **1,861,056** cases after
flip/xpose, with no counterexample, proving the selected mask is positive.

That proof and the complete `[0,2**20]` uint8-carrier proof remain applicable:
the residual candidate changes only `Gv/Gh`, the exact contraction above
reconstructs the same `e3`, and the following `G`, masks, Selu carrier, `ti`,
and output are untouched.  The directed and full-model raw audits below also
confirm every audited downstream tensor is identical.

## Four-configuration whole-model audit

Independent fresh seeds were `66212001` and `66212002`, 2,000 cases each.  All
suites ran under disable-all/default ORT with 1/4 threads.

| suite | candidate/gold per config | final raw equal | every traced tensor raw equal | errors |
|---|---:|---:|---:|---:|
| known | 266/266 | 266/266 | 266/266 | 0 |
| fresh 66212001 | 1898/2000 | **2000/2000** | **2000/2000** | 0 |
| fresh 66212002 | 1904/2000 | **2000/2000** | **2000/2000** | 0 |

The 13 traced tensors were `Gv`, `Gh`, `Gf`, `G`, `Ov`, `Oh`, `O`, `aMask`,
`bMask`, `selF`, `selLog`, `selQ`, and `ti`.  Across the four configurations,
all **17,064 case-config pairs** had raw-identical final and trace outputs.
The fresh accuracy is inherited parent behavior (94.9% and 95.2% after display
rounding); admission rests on exact pass-through, not on sample generalization.

The candidate had zero runtime errors and zero final/Gv/Gh/Gf/selLog/selQ/ti
nonfinite values.  Per configuration, the unchanged parent exposes `selF=+inf`
on 6 known, 50 first-fresh, and 50 second-fresh cases.  The candidate is raw
identical there and every downstream value and final output remains finite.

## Directed signed-zero and maximum-bound audit

An isolated `Gv/Gh` audit used 804 inputs per configuration:

- all-zero and deliberately misaligned zero cases;
- vertical and horizontal exact maxima (`2,097,150`);
- every 20x20 cyan/green basis-coordinate pair in each orientation.

Both `Gv` and `Gh` were raw-identical on 804/804 cases in each of the four ORT
configurations.  Signed-zero raw differences, nonfinite values, and runtime
errors were all zero.

## Structural gates

- ONNX full checker: pass;
- strict and data-propagating strict shape inference: pass;
- 79/79 runtime node outputs truthful in disable-all and default ORT;
- all node outputs static and resolved;
- standard ONNX domains only; no functions, nested graphs, sparse tensors,
  banned/Sequence ops, or risky lookup ops;
- largest initializer is only 60 elements; no output/example lookup was added;
- Conv-family short-bias UB findings: **0**;
- official known verification: pass.

## Artifacts

- `audit.py` — independent graph/selector/static/cost/full-model/directed audit;
- `audit.json` — compact machine-readable evidence and decision;
- `REPORT.md` — this fail-closed review record.
