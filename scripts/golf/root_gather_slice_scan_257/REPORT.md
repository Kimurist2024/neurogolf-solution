# All-400 Gather to arithmetic Slice exact scan

## Outcome

**Accepted candidates: 0. Projected gain: +0.0.**

The scan used immutable `submission.zip` SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
It inspected all 400 members, covering **406** standard `Gather` nodes in **65**
tasks.

For each one-dimensional constant index tensor, indices were normalized against
the static axis length, checked in-range and duplicate-free, and required to be
a monotone arithmetic sequence.  The exact Slice was then constructed as:

- positive step: `start=first`, `end=last+1`;
- negative step ending above zero: `end=last-1`;
- negative step including zero: `end=-(axis_length+1)`, which ONNX normalizes
  and clamps to the exclusive `-1` boundary.

The generated Slice sequence was exhaustively compared to the normalized
Gather index sequence before a node was marked convertible.  The scan covered
64 step-1 sequences, one stride-7 sequence, and one reverse step-1 sequence.
Gather and Slice output shapes were also required to match by the static shape
formula.

## Constant and combined-rewrite accounting

- 66 nodes in 22 tasks passed the static arithmetic-sequence proof.
- 35 initializer groups had no uses outside convertible Gather index inputs.
- New one-element INT64 `starts/ends/axes/steps` values were globally shared
  with existing aliases and with one another.
- The exact subset of removable index initializers maximizing parameter saving
  was selected per task.  Only task191 and task285 had positive net savings.
- task090's singleton `[0]` Gather is convertible to `Slice(0:1)` as a sequence,
  but removing one index parameter requires an equivalent replacement constant;
  projected saving is zero.  No task090 candidate was emitted.

## Candidate results

| task | exact replacement | cost | full/strict/UB0 | known-four raw | runtime shapes | decision |
|---:|---|---:|---|---|---|---|
| 191 | `[1,8] -> Slice(1:9:7)` and `[3,4] -> Slice(3:5:1)`; shared constants, params -3 | 3436 -> **3433** | PASS | disable-all 267/267 raw-identical at 1/4 threads; default session fails | **35 mismatches** | reject cloak |
| 285 | `[0,1] -> Slice(0:2:1)`; all Slice constants already aliased, params -2 | 8623 -> **8621** | PASS | disable-all 265/265 raw-identical at 1/4 threads; default session fails | **55 mismatches** | reject cloak |

Both candidates remain correct under the official-like disabled-optimization
scorer and are strictly cheaper.  Neither is eligible:

- task191 declares output `[1,10,1,1]` while runtime produces
  `[1,10,30,30]`; default ORT fails a multi-axis `CenterCropPad` shape check.
- task285 contains widespread `CenterCropPad`-hidden dimensions; default ORT
  fails a `Concat` merge (`inferred=3`, `declared=1`).

Thus the rewrites preserve the disabled-mode raw values but inherit prohibited
shape-cloak lineages.  Static full checking and strict data propagation are not
sufficient to make either model truthful.

## Fresh disposition

No candidate passed the runtime-shape and four-configuration known gates.
Consequently the strict-lower survivor set was empty and fresh
`2 seeds x 1000 x 4 configurations` was not run.  Runtime errors must be zero;
default-session construction failures are terminal rejections.

## Reproduction

```bash
.venv/bin/python scripts/golf/root_gather_slice_scan_257/scan_gather_slice.py
```

`scan_result.json` records all 406 proof attempts, normalized sequences, Slice
bounds, alias/subset accounting, hashes, costs, shape traces, and known-four raw
results.  Candidate ONNX files are retained only as rejected probes.

