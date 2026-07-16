# Giant / huge-fan-in guarantee re-audit

## Outcome

No candidate is safe to adopt. The accepted set is empty, projected gain is
`+0.0`, and no ZIP or protected score file was changed.

This lane searched 1,136 historical `LOOP_STATE` / `REPORT` / `result` /
`winner_manifest` files and revisited the 41 Wave30b payloads plus the strongest
exact/private candidates. Dedicated task198/254/267/323 lanes were excluded as
requested. The broadened task set was task013/051/064/070/199/202/328/333/379.

## Gain-ranked decisions

| task | baseline -> candidate | apparent gain | decisive evidence | decision |
|---:|---:|---:|---|---|
| 202 | 48 -> 28 | +0.538996501 | known 230/230 in all four configs, but seed 71418202 valid case 160 is wrong by 20 cells in every config | reject counterexample |
| 070 | 75 -> 64 | +0.158605030 | 0/266 known in disabled/default x threads1/4 | reject known |
| 199 | 261 -> 241 | +0.079723474 | 0/266 known in disabled/default x threads1/4 | reject known |
| 328 | 558 -> 554 | +0.007194276 | known 267/267, but retained fresh has four values in `(0,0.25)` per mode; min positive `7.31687e-11` | reject margin |
| 333 | 423 -> 421 | +0.004739345 | known 265/265 x4 and clean margin, but the 36->35 input contraction change lacks an all-support platform proof | reject guarantee |
| 013 | 638 -> 636 | +0.003139720 | known 267/267 x4, clean margin, truthful 55/55 outputs; exact constant identity holds, but 2,721,600 support states x4 were not completed | reject guarantee |
| 379 | 1949 -> 1947 | +0.001026694 | raw-equal fresh 5000/5000, but truth is only 4999/5000 | reject counterexample |

task051's factor candidate costs 280 versus the current 279. task064 has no
known-perfect retained candidate below the current 271. Neither reaches the
strict-lower gate.

## Four-configuration evidence

The independent rerun used ORT CPU in:

- `ORT_DISABLE_ALL`, threads 1 and 4;
- default optimization, threads 1 and 4.

task202/333/013 passed every convertible known case in every configuration
with runtime errors 0, nonfinite values 0, and no raw positives in `(0,0.25)`.
task070 and task199 were wrong on all 266 known cases in all configurations.
The task202 generator-valid counterexample was independently reproduced in all
four configurations, so its private-zero lineage cannot receive the user's
pass-guarantee exception.

## Closest algebraic case: task013

The removed initializer is `T_zero=[1,0]`. In the candidate,
`Qor[i,i,i,i,i]` is exactly `[1,0]` in float32, so the ONNX real-valued tensor
identity is exact. The candidate is shape-truthful, standard-domain,
lookup-free, checker/strict-data-propagation clean, and has official-like cost
`488 memory + 148 params = 636`.

However, this changes how the same tensor is presented to a 51-input floating
`Einsum`. The generator has 37,800 structural states and 72 ordered color
pairs, or 2,721,600 complete states. Because all states were not executed under
all four ORT configurations, contraction-order/platform equivalence is not a
guarantee. The fail-closed decision is therefore rejection, not a winner.

## Evidence

- `audit.json`: complete rerun details, raw margins, shapes, counterexample grids, and hashes
- `official_costs.json`: independent official-like profiler results
- `task013_runtime_shapes.json`: sanitized all-node runtime shape trace
- `result.json`: ranked dispositions
- `winner_manifest.json`: empty winner set and fixed guarantee policy

`submission_base_8005.16.zip` remains SHA-256
`73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`.
