# task071 truthful-cost audit

## Decision

**NO PROMOTION.**  The current LB authority reports cost **185**, but that
number depends on stale one-element declarations for tensors that execute with
length 30.  The cheapest fully truthful, exact equivalent produced here costs
**262** (`memory=207`, `params=55`), so it is not an improvement.

Neither `submission.zip`, `all_scores.csv`, nor `best_score.json` was changed.

## Authority finding

The pinned `submission.zip` member is SHA-256
`40f1edfc464d21e8388f3013f926ad958f374cf84b1ac67d48f740fb0afd3902`.
Its declarations and runtime shapes disagree:

| tensor | declared | runtime |
|---|---:|---:|
| `gather_u8_s` | `[1]` | `[30]` |
| `gather_i32` | `[1]` | `[30]` |
| `output` | `[1,10,30,1]` | `[1,10,30,30]` |

The official-profile-compatible local path therefore reports only
`memory=129, params=56, cost=185`.  This is not a truthful-shape baseline.

## Truthful candidate

`candidates/task071_truthful_dense.onnx` removes the identity
`Shape -> Reshape` chain, replaces `CastLike(..., i32zero)` with an explicit
`Cast(to=INT32)`, removes the now-unused initializer, and gives every tensor
its real static shape.

- SHA-256:
  `ce023f9d879c4c8d0b32dde95c130dc9db5684e45e8a5b16d79907da41c6bb9a`
- ONNX `full_check`: pass
- strict shape inference with `data_prop=True`: pass
- all 26 node outputs exposed at runtime: every declared shape equals runtime
- canonical output: `[1,10,30,30]`
- official local gold: exact
- visible minimum nonzero margin: `1.0`
- official profile: `memory=207`, `params=55`, `cost=262`
- fresh seed `7120260715`: `2000/2000`, no runtime/nonfinite/shape errors
- fresh seed `7120260716`: `2000/2000`, no runtime/nonfinite/shape errors
- combined fresh: **4000/4000 (100%)**

## Why the truthful cost rises

The exact construction ends in `Gather(input, dynamic_index, axis=3)`.  Its
truthful intermediate-cost decomposition is:

| component | bytes/elements charged |
|---|---:|
| dynamic `INT32[30]` Gather index | 120 bytes |
| compact `UINT8[30]` index precursor | 30 bytes |
| scalar/small-vector axis and side routing | 57 bytes |
| initializers | 55 elements |
| **total** | **262** |

Thus the dynamic index plus its compact precursor alone consume 150 of the
185 authority budget.  The remaining truthful logic and parameters consume
112 more.  This establishes that metadata repair or a local tail shave cannot
beat 185; a genuinely different algorithm would be required.  It is not a
claim of a global mathematical lower bound over every possible ONNX graph.

## Sparse initializer attempt

The generator is fixed to a 16x16 logical grid, so entries 16..29 of the
coordinate vector are provably multiplied by zero.  A sparse form would reduce
the parameter count from 55 to 39.  It was rejected before accuracy testing:
ONNX strict full-check/shape inference sees the sparse `Einsum` inputs as rank
zero and reports that their rank does not match the one-dimensional equation
indices.  Because strict checker and truthful shape inference are mandatory,
the sparse artifact was not retained or admitted.

## Reproduction and artifacts

Run from the repository root:

```bash
.venv/bin/python scripts/golf/focus_task071_gold/build_truthful_sparse.py
.venv/bin/python scripts/golf/focus_task071_gold/audit.py
```

- `build_truthful_sparse.py`: deterministic truthful builder plus fail-closed
  sparse experiment
- `build.json`: authority identity and build evidence
- `audit.py`: strict shape, gold, margin, and fresh-4000 audit
- `evidence.json`: machine-readable final evidence and unchanged root hashes
- `candidates/task071_truthful_dense.onnx`: fully gated reference only, rejected
  for cost
