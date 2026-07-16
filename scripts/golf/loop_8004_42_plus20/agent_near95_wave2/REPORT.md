# Near-95 Wave 2 rescreen

## Outcome

The exhaustive targeted inventory found one new policy-95 candidate for
task009. It is isolated only; this lane did not build or modify a submission
ZIP.

| task | current cost | candidate cost | projected gain | known dual | fresh dual | decision |
|---:|---:|---:|---:|---:|---:|---|
| 009 | 2619 | 2586 | +0.012680285 | 265/265 | 9530/10000 across two independent seeds | policy95 finalist |

Candidate:
`candidates/task009_b265f7f83d8f_cost2586.onnx`, SHA-256
`b265f7f83d8fbf66c9388b9edfe0111d2b77a4b610377a3994a9c483fb445d28`.

Both ORT modes produced the same results with zero runtime errors:

- initial fresh screen: 479/500 (95.8%);
- independent seed A: 4752/5000 (95.04%);
- independent seed B: 4778/5000 (95.56%);
- combined independent confirmation: 9530/10000 (95.30%).

The model passes full ONNX checking, strict shape inference with data
propagation, static positive shapes, complete known correctness in both ORT
modes, actual-cost profiling, and runtime-shape tracing. It has no
Conv-family bias finding, lookup op, giant initializer, Einsum, nested graph,
sparse initializer, custom domain, or explicit private-zero/quarantine
lineage. Its only source is `others/4/task009_improved.onnx`.

The current task009 member has prior fresh-5000 100% evidence, whereas this
candidate is 95.30%. Therefore the result is deliberately labeled policy95,
not generator-SOUND. It satisfies the user's numerical 95% threshold but
should remain isolated until the root loop explicitly chooses that risk.

## Coverage

The scanner inspected 598 unique non-baseline SHA values across every loose
ONNX and every canonical target member in 1,243 repository ZIPs:

| task | unique SHA | terminal result |
|---:|---:|---|
| 396 | 80 | cheaper leads were lookup lineage; clean graphs were not cheaper |
| 255 | 60 | no clean candidate had lower actual cost than current 1162 |
| 196 | 45 | four cheaper known-complete leads were shape-cloaked |
| 365 | 74 | clean cost1337 lead scored 457/500 (91.4%) |
| 048 | 49 | clean cost378 family scored 457/500 (91.4%) |
| 096 | 79 | cost1128 lead failed default-ORT session creation; cost1111 has QLinearConv bias UB |
| 023 | 56 | cost1497 scored 0/500; cost1541 scored 439/500 |
| 009 | 51 | cost2586 finalist passed; cost2457 scored 430/500 |
| 202 | 25 | cheaper leads used 21-input floating Einsum/private-zero lineage |
| 205 | 79 | cheaper leads were lookup, giant-Einsum, or private-zero lineage |

`rescreen.json` records every SHA, source, actual cost, structural reason,
known result, and fresh disposition. `winner_manifest.json` is the concise
handoff record. No protected root file or baseline archive was modified.
