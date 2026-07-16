# high67 audit — guaranteed task267 winner

## Decision

**SAFE WINNERS: 1** for `task071,137,183,210,234,241,267,308`.

`task267_r02_static30.onnx` is eligible under the explicit private-zero
guarantee exception.  It reduces the immutable 8005.16 task267 cost from 60 to
30, an isolated score gain of `ln(60/30) = 0.6931471805599453`.

No submission ZIP or protected root file was modified by this lane.

## Frozen payload

- immutable base: `submission_base_8005.16.zip`
- base SHA256: `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`
- winner: `scripts/golf/loop_7999_13/lane_archive_all400/task267_r02_static30.onnx`
- winner SHA256: `4ca7f921c34f87ef71512a8b680de7c984a2b42cd55b338b57aaabc012321387`
- official-like actual cost: memory 0 + params 30 = 30
- immutable task267 cost: memory 0 + params 60 = 60

## Full finite-support proof

The authoritative generator is `task_aabf363d.py`: fixed 7x7 input, creature
size `N in {12,13,14,15}` inside rows/columns 1..5, and an ordered pair of
distinct colors from 1..9.  This reduces to `4 * 9 * 8 = 288` states.

The r02 graph is one float32 Einsum.  Its equation reduces to:

```text
A_o = sum_rc X[o,r,c] * p[r]
D_o = sum_uv X[o,u,v] * p[u]^61
C_d = sum_st X[d,s,t] * p[s]^7
Y[o,h,w] = A_o * D_o * sum_d(C_d * X[d,h,w])
```

All creature rows 1..5 have the identical coefficient `p=0.05`, so every
reduction depends only on `N` and the two colors.  Columns and exact connected
placement do not affect a coefficient; the final pointwise `X[d,h,w]` factor
transfers arbitrary creature occupancy.  Thus the 288 reduced states cover the
generator's full finite support, not merely sampled shapes.

All 288 states passed in each of four runtime configurations:

| threads | ORT mode | correct | runtime errors | nonfinite | near margin | min positive | max abs raw |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | DISABLE_ALL | 288/288 | 0 | 0 | 0 | 775.5849 | 1.40932e13 |
| 1 | ENABLE_ALL | 288/288 | 0 | 0 | 0 | 775.5849 | 1.40932e13 |
| 4 | DISABLE_ALL | 288/288 | 0 | 0 | 0 | 775.5849 | 1.40932e13 |
| 4 | ENABLE_ALL | 288/288 | 0 | 0 | 0 | 775.5849 | 1.40932e13 |

The source-channel suppression is also bounded: `D_source =
N*(float32(0.05)^61)`, whose magnitude is below the smallest float32 subnormal
and necessarily rounds to positive zero.  True-positive logits retain a margin
above 775.

## Structural and known gates

- known corpus: 264/264 in DISABLE_ALL and 264/264 in default mode; runtime 0,
  near-margin 0, truthful output `[1,10,30,30]`.
- full ONNX checker and strict shape inference with data propagation: pass.
- all dimensions static and positive; runtime shape mismatch/shape cloak: 0.
- standard domain only; banned ops, nested graphs, lookup/scatter, Conv bias,
  sparse initializer, and nonfinite initializer/output findings: 0.
- the sole graph node is a 73-input Einsum.  It is admitted only through the
  user's full-support guarantee exception; unlike r01/r03, r02 has the lowest
  raw magnitude (`1.409e13`) and a large positive margin.

## Other targets

The retained inventory was fully audited.  Baseline costs were task071 188,
task137 256, task183 160, task210 16, task234 368, task241 0, task267 60, and
task308 434.  Task071's lower leads failed complete known gold (best 264/265),
task183's cost-91 lead failed 0/265, and no other target had a strict-lower
complete-known lead.  The broader loose scan was stopped on root instruction
once the guaranteed task267 winner was proven.

## Evidence

- `history_lead_audit.json`: all retained leads and official-like costs.
- `task267_exhaustive.json`: frozen hashes, structure, dual known corpus, full
  288-state x four-runtime audit, raw margins, and algebraic support proof.
- `winner_manifest.json`: the single integration candidate.

