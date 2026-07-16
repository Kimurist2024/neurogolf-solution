# task066 residual exact regolf — lane 208

## Verdict

**ADMIT cost 551 candidate; not merged by this lane.**  Starting from the newly
staged cost-561 task066 parent, this candidate removes the ten-parameter
`greenhalf10` initializer by reconstructing the same green selector inside the
two existing `Gv/Gh` Einsums from already-live initializers.

- Parent: `others/71407/task066.onnx`
- Parent SHA256: `2e3bd402f667062b32858d3a11182d3e8050d833d2974d1d37fbadd688f4648b`
- Candidate: `task066_residual_cost551.onnx`
- Candidate SHA256: `622b3b28271806949bb18e8b9517335d49cb0383410caf36a19e064d95798dd3`
- Parent official profile: memory 346, params 215, **cost 561**
- Candidate official profile: memory 346, params 205, **cost 551**
- Incremental score gain: `ln(561/551) = +0.01798609636978192`
- Combined gain over the pre-Selu cost-562 authority:
  `ln(562/551) = +0.019767040740776582`

The root submission/CSV and `others/71407` were not edited.  Completion hashes
remain `submission.zip=4eb324d7...`, `all_scores.csv=8c99379c...`, and staged
task066 `2e3bd402...`.

## Exact green-selector reconstruction

The parent initializer is the float32 vector `greenhalf10=e3`, selecting color
3.  Existing `Uchan` and `Vchan` rows satisfy:

```text
Uchan[0] * Vchan[2]       = -e3
dot(Uchan[2], Vchan[2])   = -1
(-e3) * (-1)              =  e3
```

Existing selector tensors choose those rows without adding a node or parameter:

```text
sum_j Trow[a,j]*Tcol[a,j] = [1,0,0]   # choose row 0
sum_k Tcol[e,k]*z1[k]     = [0,0,1]   # choose row 2
```

Nodes 22 and 23 therefore replace their `greenhalf10[d]` operand with the
following in-Einsum contraction:

```text
Uchan[a,d] * Vchan[e,d]
* Trow[a,j] * Tcol[a,j]
* Tcol[e,k] * z1[k]
* Uchan[f,l] * Vchan[f,l]
* Tcol[f,m] * z1[m]
```

All factors and partial sums are in `{-1,0,1}`, so the selector identity is
exact in float32.  The reconstructed vector differs at unused zero entries
only in signed-zero representation (`-0 == +0`); complete whole-node tracing
below confirms `Gv` and `Gh` are raw-bit identical in every audited case and
ORT configuration.

The protobuf whitelist comparison found no other mutation:

- changed nodes: exactly `Gv` and `Gh` at indices 22 and 23;
- removed initializers: exactly `greenhalf10`;
- all 75 other nodes and all 17 common initializers: protobuf-identical;
- graph/model shell, I/O, value-info, opsets, and metadata: identical.

Because the candidate produces identical `Gv/Gh`, the independently established
`G=2*cyan_bitmask`, 1,861,056-case geometry proof, `selF>=1` proof, and complete
`[0,2**20]` downstream carrier proof from review 206 remain unchanged.

## Four-configuration whole-model audit

Independent fresh seeds were `66208001` and `66208002`, 2,000 cases each.  Every
suite was run under disable-all/default ORT with 1/4 threads.

| suite | candidate/gold per configuration | final raw equal | `Gv/Gh` raw equal | all traced downstream raw equal | errors |
|---|---:|---:|---:|---:|---:|
| known | 266/266 | 266/266 | 266/266 | 266/266 | 0 |
| fresh 66208001 | 1893/2000 | **2000/2000** | **2000/2000** | **2000/2000** | 0 |
| fresh 66208002 | 1886/2000 | **2000/2000** | **2000/2000** | **2000/2000** | 0 |

The traced downstream set was `Gf`, `G`, `Ov`, `Oh`, `O`, `aMask`, `bMask`,
`selF`, `selQ`, and `ti`, in addition to `Gv/Gh` and the final output.  Every
one was raw-identical to the cost-561 parent for every audited input and mode.

The parent's arbitrary-fresh correctness is inherited (`3779/4000 = 94.475%`),
but this candidate is exact pass-through on all 4,000 fresh cases in all four
configurations.  Runtime errors and final nonfinite values were zero.  The
known inherited `selF=+inf` cases remain raw-identical; `selQ`, `ti`, and final
output had no nonfinite values.

## Structural gates

- official profile: **551 = memory 346 + params 205**;
- ONNX full checker: pass;
- strict shape inference: pass;
- strict shape inference with data propagation: pass;
- every node output statically resolved: pass;
- runtime shape/dtype trace: **79/79 truthful** in disable-all and default ORT;
- standard ONNX domains only; no functions, nested graphs, sparse initializers,
  banned ops, or Sequence ops;
- Conv-family short-bias UB findings: **0**;
- known gold in all four ORT configurations: 266/266;
- runtime errors: 0; final nonfinite: 0.

## Rejected residual families and local floor

All rejected binaries are quarantined under `REJECTED_DO_NOT_MERGE/`.

| attempted family | nominal cost | result |
|---|---:|---|
| feed uint8 `cRplus` directly to OneHot | 543 | checker accepts, but ORT CPU has no uint8-index OneHot kernel |
| replace `forceB/useB` with bool-data `Where(force,hasB,noA)` | 547 | support truth table is valid, but ORT CPU has no bool-data Where kernel |
| assume canonical `cOut=cR+1` and remove direction branch | 548 | **reject**: known case index 3 has reversed explicit marker layout; 265/266 |
| OneHot index Cast to int8/uint8/int16/uint16/uint32/f16/f32 | 544–548 | no ORT OneHot kernel for these index types |
| OneHot index Cast to int64 | 555 | works and is exact, but is dominated by int32 cost551 |

The support relation allows three `(noA,hasB)` states: `(1,1)`, `(0,0)`, and
`(0,1)`.  `useB` is their majority with `force_any`; no single supported binary
boolean operator realizes all three rows.  The existing `And` + `Or` is thus a
local two-output floor after bool `Where` is excluded.

All four force predicates are necessary over legal geometry: each occurs alone
on some support tuple.  The interval predicates use uint8 subtraction wrap to
encode `[10,12]` and `[15,18]` in two nodes each; replacing them with two-sided
comparisons adds nodes.  Remaining scalar initializers are shared or cost one
parameter, while deriving them through runtime Shape/arithmetic costs at least
one counted output and gives no strict decrease.  Attribute/rank/axes deletions
do not change the official parameter or memory cost.  Under these audited local
families, **551 is the residual structural floor**.

## Artifacts

- `task066_residual_cost551.onnx` — accepted lane candidate.
- `build_candidate.py` — source-hash-bound builder and index-dtype probes.
- `audit_candidate.py` — exact selector/delta proof, four-config known/fresh
  trace, official profile, static/UB, and runtime-shape gates.
- `REJECTED_DO_NOT_MERGE/` — failed or dominated experimental binaries.

