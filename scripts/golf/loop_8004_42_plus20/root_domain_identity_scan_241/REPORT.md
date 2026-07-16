# All-400 exact domain-aware arithmetic identity scan

## Outcome

No admissible candidate was found. Four models were strictly cheaper under the
static official profile, but every one failed a required runtime or truthful
shape gate. No fresh run or staging action was performed.

Authority: `submission_base_8009.46.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
The root `submission.zip` had the same SHA at final verification. Neither file
was modified by this lane.

## Census

All 400 current authority members were parsed. Full strict shape inference with
data propagation passed for 393 and failed for 7 (tasks 018, 112, 117, 170,
243, 245, and 397). Exact counts of the requested operator families were:

| family | count |
|---|---:|
| `Pow` | 18 |
| `Add` / `Sub` / `Mul` / `Div` | 518 / 617 / 400 / 394 |
| `Abs` / `Relu` / `LeakyRelu` / `Selu` | 10 / 1 / 11 / 75 |
| `Clip` / `Min` / `Max` | 15 / 55 / 51 |
| `Equal` / ordered comparisons | 471 / 434 |
| `ReduceSum` / `Mean` / `Max` / `Min` | 33 / 3 / 74 / 22 |
| `ReduceL1` / `L2` / `LogSum` / `LogSumExp` / `SumSquare` | 46 / 6 / 9 / 1 / 3 |

The per-node census in `scan.json` records task, node index, attributes,
initializer values, inferred input/output metadata, and any exact identity
proof.

Additional exact screens:

- scalar `Pow` exponent exactly 0, 1, or 2: 0 nodes;
- `Selu` with `gamma=1` and a provably nonnegative input: 0 nodes;
- comparison result fixed over the complete integer dtype domain: 0 nodes;
- neutral scalar arithmetic connections: 5, all retained because they create a
  broadcast rank or reuse a still-live shared scalar, so bypassing them does not
  strictly lower cost;
- unsigned `Clip` inputs with a redundant lower bound of zero: 5, but every zero
  initializer remains shared and removing only the optional bound leaves node
  memory and parameter cost unchanged;
- scalar initializers inspected: 1,403. Only one pair was identical in shape,
  dtype, and bytes: task264 `s1` / `axes1`;
- three `LeakyRelu` nodes in task243 are positive-branch identities because
  each directly follows an unsigned-to-float cast. Task243 fails strict data
  propagation (`Reshape` 30 versus declared 1), so no admissible model may be
  derived from that source under this brief.

## Strict-static probes and decisive rejection

| task | exact rewrite | static cost | decisive gate |
|---:|---|---:|---|
| 264 | alias bit-identical `axes1` to `s1` | 344 -> 343 | Disabled-ORT known raw 265/265 in threads 1/4, but default ORT session creation fails and runtime trace has 44 declared/actual shape mismatches. |
| 366 | bypass four declared-singleton `ReduceMax` nodes | 7987 -> 7507 | The declared `[1]` sources execute as `[15]`; the singleton premise is false at runtime. |
| 377 | bypass declared-singleton `ReduceMin` | 409 -> 407 | Disabled-ORT raw/correct only 159/266; default authority errors 266/266 and shape trace is not truthful. |
| 388 | bypass declared-singleton `ReduceMin` | 85 -> 83 | Runtime source `background` is `[1,1,3,3]`, not declared `[1,1,1,1]`; candidate errors 266/266 in every configuration and has 13 shape mismatches. |

Task319 also had a declared-singleton `ReduceMax` observation, but its actual
profile increased from 979 to 1901 after bypassing it, proving the inferred
singleton was not a valid cost identity. It was rejected before writing a
strict-lower result.

Thus the scan found eight exact-static observations: seven singleton reduction
bypasses and one initializer alias. Four appeared strict-lower under declared
shapes, and zero survived runtime/error0/truthful-shape validation.

## Validation policy

For every static-lower probe the lane required:

1. full ONNX checker and strict shape inference with `data_prop=True`;
2. official runtime `memory + parameters` profiling;
3. declared-versus-runtime tracing of every typed node output;
4. raw byte equality, threshold equality, correctness, nonfinite0, and error0
   in disabled/default ORT with threads 1 and 4 over all known cases;
5. only after those gates, two independent fresh generator seeds in all four
   configurations.

No probe passed steps 3--4, so the fresh gate was intentionally not entered.
The ONNX files under `candidates/` are marked rejected reproduction probes and
must not be staged.

## Evidence and reproduction

- `scan.json`: complete all-400 census, static costs, SHAs, and final
  admissibility dispositions.
- `audit.json`: known four-configuration raw/error results and runtime shape
  traces for tasks264, 377, and388.
- `candidates/README.md`: explicit non-stageable status of probe files.

```bash
.venv/bin/python scripts/golf/loop_8004_42_plus20/root_domain_identity_scan_241/scan_domain_identities.py
.venv/bin/python scripts/golf/loop_8004_42_plus20/root_domain_identity_scan_241/audit_candidates.py
```

The NeuroGolf scoring skill materially shaped the fail-closed decision: a
declared singleton can appear to reduce charged intermediate memory while its
runtime tensor is larger. Runtime shape truth therefore overrode every apparent
static gain.
