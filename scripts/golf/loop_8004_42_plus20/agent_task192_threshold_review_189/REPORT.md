# task192 threshold k31 independent review 189

## Verdict

**REJECT / DO NOT MERGE.**  `task192_hardsigmoid_k31.onnx` is structurally clean and
passes every known example, but it is not equivalent to the staged ArgMax+OneHot model
on the generator domain. One independent fresh stream contains 5 failures in 5000 cases.

The candidate has aggregate fresh accuracy `9995/10000 = 99.95%`, but this review is
fail-closed and requires both 5000-case streams to be perfect, with exact raw and sign
equality to the staged authority. A sampled 99.95% result cannot establish the requested
private-safe equivalence.

## Official cost

All three profiles were produced by the competition-equivalent `score_and_verify` path
with complete known correctness required.

| model | memory | params | cost | known-correct |
|---|---:|---:|---:|---:|
| immutable 8009.46 task192 | 88 | 1521 | 1609 | yes |
| staged ArgMax+OneHot | 208 | 941 | 1149 | yes |
| HardSigmoid k31 candidate | 200 | 938 | 1138 | yes |

The candidate saves 11 cost versus the staged model and 471 versus the immutable model.
Projected task-score gains are respectively `+0.0096196632` and `+0.3463405323`.

## Structural and runtime gates

- ONNX full checker: PASS
- strict shape inference with `data_prop=True`: PASS
- truthful runtime shapes: PASS, `4/4` node outputs matched their declarations
- standard domain/opset only: PASS (`ai.onnx` opset 18)
- ops: `Einsum×2`, `HardSigmoid×1`, `Concat×1`
- lookup/index ops: 0
- `Hardmax`: 0; `HardSigmoid` is a different standard operator
- banned ops, nested graphs, functions, sparse initializers: 0
- Conv/ConvTranspose/QLinearConv bias UB findings: 0
- nonfinite initializers and traced values: 0
- all intermediate shapes static and positive; official candidate cost `200+938=1138`

## Known corpus, four configurations

The known corpus contains 265 cases. Candidate and staged model were compared in all four
configurations: optimizations disabled/default × 1/4 threads.

Every configuration produced:

- candidate correctness `265/265`
- staged correctness `265/265`
- candidate/staged raw-bitwise equality `265/265`
- candidate/staged sign equality `265/265`
- runtime errors 0, nonfinite values 0

Known color counts explain why the replacement appears valid on visible data: dominant
color `37..169`, runner-up `5..26`, and the number of channels selected by k31 is always 1.

## Fresh two-seed audit

Fresh comparison used ORT_DISABLE_ALL, one thread, independent Python RNG seeds. The
readable task192 rule reproduced generator output in all `10000/10000` cases, and the
staged ArgMax+OneHot model was correct in all `10000/10000` cases.

| seed | candidate right | staged right | raw equal | sign equal | runtime errors | nonfinite | dominant count | runner-up count | k31 selected channels |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 192189031 | 4995/5000 | 5000/5000 | 4995/5000 | 4995/5000 | 0 | 0 | 27..219 | 2..31 | 0..1 |
| 192189977 | 5000/5000 | 5000/5000 | 5000/5000 | 5000/5000 | 0 | 0 | 33..203 | 2..29 | 1..1 |

The first mismatching case is seed `192189031`, index 342, grid `13×18`, with 66 wrong
decoded cells. The first stream has exactly five cases where no histogram channel reaches
32. In those cases the true box color is still the unique ArgMax, but
`HardSigmoid(hist, alpha=1, beta=-31)` returns an all-zero selector because every count is
at most 31. The staged ArgMax+OneHot selector remains correct.

This is a rule-level counterexample, not an ORT, shape, margin, or numerical-stability
failure. Candidate positives remain at least 1, maximum nonpositive is 0, and no values
occur in `(0, 0.25)`.

## Integrity

- candidate SHA-256: `91315f9982649a65341134c541f904dc5398168600475a4d4f916b09b2f41941`
- staged SHA-256: `19fbdce89a5c89f5ff376b2fbbdb630ead5535d5ed5ebe7d9914a4de89e5023c`
- immutable/root ZIP SHA-256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `all_scores.csv` SHA-256: `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`
- hashes matched before and after the audit; root and other stages were not modified

Machine-readable evidence is in `result.json`; the reproducible audit is
`audit_threshold.py`.
