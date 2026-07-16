# Tasks 070 / 254 / 267 residual exact audit

## Decision

No strict-lower exact candidate remains for any of the three LB8009.46
authorities. The result is **NULL for task070, task254, and task267**; aggregate
gain is 0 and no model is emitted.

Authority ZIP:

- `submission_base_8009.46.zip`
- SHA-256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`

| Task | Member SHA-256 | Nodes | Initializers | Memory | Params | Cost | Result |
|---:|---|---:|---:|---:|---:|---:|---|
| 070 | `a45fe09083c363ab9aae49de2497c55356a3d5bfab324ec2ab6b6ed949cd1c92` | 1 | 4 | 0 | 66 | **66** | null |
| 254 | `814ece451a8f8eda8e9221d58e2f4fb3359fa396dfe971f6ad97693f453b15f8` | 1 | 6 | 0 | 42 | **42** | null |
| 267 | `4ca7f921c34f87ef71512a8b680de7c984a2b42cd55b338b57aaabc012321387` | 1 | 1 | 0 | 30 | **30** | null |

All three consist of one standard-domain Einsum whose only node output is the
free graph output. Each passes the full ONNX checker and strict shape inference
with data propagation, declares and infers truthful `[1,10,30,30]` input/output
shapes, has no stored `value_info`, and has no dead initializer. Their official
cost is therefore already exactly their initializer element count; no memory
shave exists.

## Generator boundary

- **task070 / `32597951`**: a 17x17 blue/background tiling is overlaid by a
  2–10 by 2–10 cyan rectangle; blue cells inside that rectangle become green
  in the output. The generator includes a fully randomized 17x17 source tile,
  so a new lower graph requires authority raw equivalence or a true complete
  support proof. No approximate/private-risk lead was admitted.
- **task254 / `a61f2674`**: fixed 9x9 gray bars at alternating columns; the
  unique shortest bar becomes red and the unique tallest becomes blue. Its
  complete reachable support has 21,168 ordered parameter tuples. The current
  cost-42 authority already has prior exhaustive evidence for all 21,168 cases
  in four ORT configurations (84,672/84,672, error0, nonfinite0).
- **task267 / `aabf363d`**: a 12–15-cell creature in the interior of a fixed
  7x7 grid is recolored using the marker color at `(6,0)`. The current cost-30
  authority depends only on creature size, the two ordered distinct colors,
  and pointwise occupancy. Its prior complete reduced-support proof covers
  `4 * 9 * 8 = 288` states in four ORT configurations (1,152/1,152, error0,
  nonfinite0).

## Exact residual analysis

### task070

The current 66-parameter member is already the exact initializer fusion that
replaced `U[3,3]` plus `D[3,2,2,2]` (33 elements) by one
`[3,2,2,2]` tensor (24 elements), saving 9 from the previous cost-75 member.
The remaining initializer sizes are `R=30`, `T=6`, `C=6`, and fused `U/D=24`.

Every individual factor is exact-mode-rank full: `R` has ranks `(3,3)`, `T`
and `C` have `(2,2)`, and the fused tensor has `(3,2,2,2)`. The only residual
precontract family is the three identical `R x T` contractions. A dense
precontract has 20 elements, but `R` has six total uses and must remain;
replacing `T[2,3]` therefore changes `R + T = 36` to `R + RT = 50`, **+14**.

Historical lower leads do not survive: the cost-64 latent-component removal
fails known output (the audited lead is 0/266 in all four configurations), and
the archived cost-50/52/53/56/58 variants each have generator-fresh failures.
The alternative fused member is cost 66, a tie.

### task254

Initializer use counts are `V:10`, `Ccoef:3`, `S0:6`, `S1:6`, `A:1`, and
`B:1`; nothing is dead or duplicated. `V`, `S0`, `S1`, `A`, and every mode of
`B` are exact-rank full. The residual precontract scan finds only local
opportunities involving shared `V`:

- `V[2,10] x S0/S1[2,2] -> [10,2]`: the 20-element `V` remains, so replacing
  a 4-element map by a 20-element product is **+16**;
- `A[2,2] x V[2,10] -> [2,10]`: `V` remains, so replacing `A` is also **+16**.

The dedicated safe rebuild already tested 60 under-budget tensor-train
candidates; none solved a complete known case, while exact rank analysis gives
an exact TT-family floor of 114 parameters. The former cost-64 15-operand
precontract loses the coupled latent state and scores 0/265. Thus the already
exhaustively proven cost-42 authority has no exact residual shave.

### task267

The authority has one initializer, `p[30]`, used 69 times by the sole Einsum.
Its 30-length row dimension is tied directly to the 30-row input indices;
truncating its 23 zero entries makes the contraction dimension invalid. An
exact sparse reconstruction stores seven nonzeros but materializes a counted
30-element float32 tensor: memory 120 + params 7 = cost 127, versus 30.
There is no initializer alias, pair to precontract, dead value, or lower exact
factor.

## Cross-cutting cleanup

- The all-authority optimizer/fusion pass changes none of the three graphs.
- Default-input removal finds no Dropout, Clip, or Reshape site.
- Sparse reconstruction costs 145/163 for task070 targets, 109 for task254's
  `V`, and 127 for task267's `p`, all above authority.
- Dtype narrowing cannot reduce parameter cost because parameters are counted
  by element, not byte; Einsum also requires a common input dtype, so a cast
  would add counted memory.
- No candidate passed the prerequisite `candidate cost < authority cost`.
  Consequently winner-only known/fresh four-configuration runs were not
  started. This is a floor rejection, not a promotion.

Root submissions, score ledgers, staging, and immutable authority artifacts
were not modified.

