# task344 POLICY90 independent review

## Verdict

**ADMIT_POLICY90** for the quarantined cost-132 candidate.

This is deliberately not an exact/SOUND admission. The candidate and the LB-white
cost-137 authority both have reachable truth counterexamples. The candidate is,
however, a normal-task POLICY90 candidate: it reaches **19962/20000 = 99.81%**
exact-example accuracy on two new large generator streams in every one of four
CPU ORT configurations. It has no runtime error, nonfinite value, shape mismatch,
lookup, shape cloak, banned/custom op, or identified UB. The 95% Wilson lower
bound from this audit is 99.7393%, comfortably above the requested 90% policy.

No file was staged or merged. The quarantined source and immutable authority were
read only.

## Identity, official cost, and gain

| item | SHA-256 | bytes | official memory | params | cost |
|---|---|---:|---:|---:|---:|
| LB8009.46 authority `task344.onnx` | `05bedf3ca834aadfc973c00fc91cafdb4d0ae1aaab374115d924e2e33fb1bf6c` | 1205 | 0 | 137 | 137 |
| quarantined candidate | `c5272a42bee419008a15d14bea734a6fb15956a863ad8e702deac0f02fcea5f6` | 1203 | 0 | 132 | 132 |

The authority archive is `submission_base_8009.46.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
The exact score gain is:

```text
ln(137 / 132) = 0.03717900324175403
```

The local official-equivalent profiler also reports known-gold correctness for
both payloads.

## Exact generator semantics

Authority source: `inputs/arc-gen-repo/tasks/task_d90796e8.py`, SHA-256
`e755ebf0dcebb7cbe7587615e6a0592c19a7974119365fde5869c91dea86aabf`.
The compact solver is `inputs/sakana-gcg-2025/raw/task344.py`, SHA-256
`224f92da3fa0c6d40111104cbe4580655002467957901404113ce11398185d92`.

The generator does the following:

1. Choose width and height independently and uniformly from 3 through 10.
2. Draw each visible gray-5 cell with Bernoulli probability 0.04.
3. Scan a one-cell-padded grid for Bernoulli-0.08 red-2 centers. Greedily accept
   a center only if its Manhattan distance from every earlier accepted center is
   greater than two.
4. Draw the red center. With probability 3/4, draw one green-3 cell in a uniformly
   chosen cardinal direction; with probability 1/4, draw no green. Padded draws
   outside the visible grid are clipped.
5. Extract the visible nonzero bitmap row-major. For every green-3 input cell,
   scan right/down/left/up in mutable output; each adjacent red-2 becomes black-0,
   and the green becomes cyan-8 if it touched a red.

The center-spacing rule guarantees that, on generator support, every visible red
has at most one adjacent green and every green has at most one adjacent red.
Therefore the mutable loop is exactly the simultaneous local rule:

```text
3 with an orthogonally adjacent 2 -> 8
2 with an orthogonally adjacent 3 -> 0
everything else                   -> unchanged
```

The one-line solver implements the four orientations by rotate-and-replace.
The generator, solver, and readable rule agree on all 266 known examples and on
all 20,000 newly generated examples. Pair-degree violations were zero.

## Private-zero, lookup, and cloak status

- task344 is absent from `docs/golf/private_zero_tasks.md` and is not a confirmed
  private-zero or unsound-incumbent task; the cost-137 authority is LB-white.
- The candidate is a numerical local-rule factorization, not fixture or coordinate
  lookup. Its only op is standard-domain `Einsum`; lookup/scatter ops are absent.
- There is no shape cloak. Strict inference with data propagation declares input
  and output as float `[1,10,30,30]`, and every runtime observation in all four
  configurations is exactly `[1,10,30,30]`.
- There are no intermediate node outputs: the sole node writes the graph output,
  which explains official memory 0. The candidate has 71 `Einsum` inputs and
  four finite float32 initializers: `V[4,10]`, `B[2,30]`, `G[4,4]`, `M[4,4]`.
- Full checker and strict shape inference/data propagation pass. Functions,
  subgraphs, sparse initializers, custom domains, sequence/banned ops, Conv-family
  bias UB, nonfinite initializers, runtime errors, and nonfinite outputs are all 0.

The candidate absorbs the authority's `H.T @ S @ H` factors into serialized
float32 `G`; it is an approximate reparameterization, not a new task lookup.

## Complete known and fresh results

Environment: Python 3.12.7, ONNX 1.21.0, ONNX Runtime 1.24.4,
`CPUExecutionProvider`, single-threaded sessions. The four configurations were
`ORT_DISABLE_ALL`, BASIC, EXTENDED, and ENABLE_ALL.

Every configuration produced the same thresholded results:

| corpus | candidate | authority | candidate wrong cells | authority wrong cells | candidate-authority sign differences |
|---|---:|---:|---:|---:|---:|
| known train+test+arc-gen (3+1+262) | 266/266 | 266/266 | 0 | 0 | 0 |
| fresh seed 344902091 | 9981/10000 | 9981/10000 | 38 | 38 | 0 |
| fresh seed 344902173 | 9981/10000 | 9981/10000 | 37 | 37 | 0 |
| fresh combined | **19962/20000 (99.81%)** | **19962/20000 (99.81%)** | 75 | 75 | 0 |

Both seeds were new and disjoint from the historical `344171501/344171777`
streams. All dimensions 3..10 and all supported colors `{0,2,3,5}` occurred.
Every row in the table has runtime errors 0, nonfinite outputs 0, and only the
truthful runtime shape `[1,10,30,30]`.

Raw margins were also identical across optimization configurations. For the
candidate:

| corpus | minimum positive true-channel raw | minimum true-channel raw | maximum false-channel raw | values in `(0,0.25)` |
|---|---:|---:|---:|---:|
| known | 17.6396484375 | 17.6396484375 | 0.0 | 0 |
| seed 344902091 | 17.1394748688 | -187.3176727295 | 163.9418640137 | 0 |
| seed 344902173 | 2.4875788689 | -171.2295227051 | 164.9873962402 | 0 |

Thus the fresh failures are not numerical threshold flicker: they are large-margin
approximation errors shared with the authority. The maximum candidate-authority
raw deltas were 0.0052490 on known, 0.0065918 on the first fresh seed, and
0.0068359 on the second.

For determinism, candidate and authority were each run five times on six cases
(known endpoints, both fresh streams, a fresh failure, and the reachable
authority-divergence witness) under every configuration. Each case/model/mode had
one raw-output hash, and each case/model also had one raw hash across all four
configurations.

## Counterexamples and comparison with truth

The first new candidate truth counterexample is seed `344902091`, index 368. At
the isolated red-2 cell `(9,0)`, truth keeps color 2. Candidate and authority both
emit background instead:

```text
candidate ch0 = +152.5676116943, ch2 = -150.1993103027
authority ch0 = +152.5674743652, ch2 = -150.1991577148
```

This produces two one-hot mask mismatches and reproduces identically in all four
ORT configurations. It proves that neither model is generator-exact.

The previously known generator-reachable candidate/authority divergence is also
reproduced in all four configurations at channel 8, row 5, column 5:

```text
authority = +0.0002454968926   (false positive versus truth)
candidate = -0.0001138478474   (correct versus truth)
```

So that witness refutes bit/sign equivalence to authority, but it is not a
candidate truth failure; the candidate improves the authority on that cell.

## Decision rationale

Exact admission is impossible because reachable counterexamples exist. Rejection
is not required by the requested policy: task344 is a normal, LB-white task rather
than a private-zero/error task, the candidate is structurally and operationally
safe, and independently measured generator accuracy is 99.81% in every requested
runtime configuration. Therefore the correct policy result is **ADMIT_POLICY90**.

