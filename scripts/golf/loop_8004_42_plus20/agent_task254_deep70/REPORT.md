# task254 finite-support guarantee audit

## Result

`task254_r01_static42.onnx` is a **verified winner** under the user's explicit
finite-support guarantee exception. It reduces official profiled cost from 76
to 42 and yields a projected score gain of **+0.5930637220**.

Decision: `ADMIT_PENDING_LB`. No root submission, integration ZIP, CSV, score
ledger, or canonical model was modified.

## Frozen payload

- Candidate: `scripts/golf/loop_7999_13/lane_b12/candidates/task254_r01_static42.onnx`
- Candidate SHA-256: `814ece451a8f8eda8e9221d58e2f4fb3359fa396dfe971f6ad97693f453b15f8`
- Candidate file size: 652 bytes
- Baseline: `submission_base_8005.16.zip::task254.onnx`
- Baseline member SHA-256: `f3473850a0a2bbe525434cab0647292c816ec3f3f39399a169799897eb211936`
- Baseline ZIP SHA-256: `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`

## Generator support proof

The authoritative generator is
`inputs/arc-gen-repo/tasks/task_a61f2674.py`, SHA-256
`a5b39b152ef64a04120567c79a8880f6d43636bdc484790d2d4c26c1287a0927`.
Its default `generate()` path fixes `size=9`, draws `offset` from `{0,1}`, and
samples distinct ordered heights from `1..9`:

- `offset=0, num=4`: `P(9,4) = 3,024`
- `offset=0, num=5`: `P(9,5) = 15,120`
- `offset=1, num=4`: `P(9,4) = 3,024`

Thus the complete reachable support is exactly **21,168 parameter tuples**.
The audit directly enumerated every tuple. No random sampling, visible-example
fit, or equivalence-class extrapolation was used.

The decoded rule matches the generator: gray vertical bars occur at
`2*idx+offset`; all cells in the unique shortest bar become red and all cells
in the unique tallest bar become blue, while the rest of the 9x9 output is
black. Distinct sampling guarantees unique minima and maxima.

## Exhaustive runtime proof

Every one of the 21,168 inputs was run in all four configurations:

1. ORT 1.24 CPU, `ORT_DISABLE_ALL`, intra/inter threads 1
2. ORT 1.24 CPU, `ORT_DISABLE_ALL`, intra/inter threads 4
3. ORT 1.24 CPU, `ORT_ENABLE_ALL`, intra/inter threads 1
4. ORT 1.24 CPU, `ORT_ENABLE_ALL`, intra/inter threads 4

Combined result: **84,672/84,672 exact threshold masks**, wrong 0, runtime
errors 0, nonfinite cases/elements 0, and output shape
`[1,10,30,30]` on every run. Each configuration independently scored
21,168/21,168.

The weakest raw positive output across the complete support and every mode was
`1.0000171661376953`; the largest false-cell raw value was exact `0.0`.
There were zero cells in the open interval `(0, 0.25)`. Maximum absolute raw
magnitude was only `525.1455688476562`. The 265-case known corpus also passed
265/265 separately in every configuration with runtime0/nonfinite0/near0.

## Structural and cost gates

The candidate passes `onnx.checker(..., full_check=True)`, strict shape
inference with `data_prop=True`, static positive dimensions, standard-domain
opsets, truthful inferred/runtime output shape, no nested graphs, no functions,
no sparse initializers, no banned/Sequence operators, finite initializers, and
lookup0. It has one standard-domain `Einsum` node and no counted intermediate
output. Official-like profiling gives memory 0, parameters 42, cost 42.

The one `Einsum` has 33 inputs. This would normally be rejected as a giant
contraction, but it contains no undefined operation and is admitted here only
because the user explicitly allowed such candidates when the finite private
support can be guaranteed. The complete parameter support, four ORT execution
configurations, raw sign margin, runtime behavior, and output shape were all
exhaustively verified. No coefficient repair was needed.

## Evidence

- `exhaustive_audit.py`: deterministic support enumerator and auditor
- `exhaustive_audit.json`: complete machine-readable structural, known-corpus,
  cost, per-mode support, raw-margin, and witness evidence
- `result.json`: concise decision record
- `winner_manifest.json`: integration handoff (candidate only; no ZIP mutation)
