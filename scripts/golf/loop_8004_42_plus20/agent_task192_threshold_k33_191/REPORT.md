# task192 HardSigmoid k33 — POLICY90 audit 191

## Verdict

**`POLICY90_ADMISSIBLE`** under the user's explicit rule for normal candidates.
The candidate exceeds 90% independently in both 5000-case streams and has zero runtime
errors and zero nonfinite values. This lane made no stage or root change.

This is not an exact or all-generator-support result. The candidate is explicitly labeled:

> `all-support=false / POLICY90 only`

## Cost and structure

Competition-equivalent `score_and_verify`, with complete known correctness required:

| model | memory | params | cost | known-correct |
|---|---:|---:|---:|---:|
| immutable 8009.46 task192 | 88 | 1521 | 1609 | yes |
| staged ArgMax+OneHot | 208 | 941 | 1149 | yes |
| HardSigmoid k33 | 200 | 938 | 1138 | yes |

The candidate saves 11 cost versus stage and 471 versus immutable task192. Projected
task-score gains are `+0.0096196632` and `+0.3463405323`, respectively.

Structural gates all pass:

- full checker and strict `data_prop=True`: PASS
- truthful runtime shapes: `4/4`, with no mismatch or nonfinite traced value
- standard ONNX opset 18; ops are `Einsum×2`, `HardSigmoid`, `Concat`
- lookup/index ops, `Hardmax`, banned ops, nested graphs, functions, sparse initializers: 0
- Conv-family bias UB findings: 0
- all node outputs static; official cost is `200 memory + 938 params = 1138`

`HardSigmoid` is a standard threshold operator and is not `Hardmax`.

## Known four-configuration audit

Across optimizations disabled/default × threads 1/4, every configuration produced:

- candidate correctness `265/265`
- staged correctness `265/265`
- raw equality to stage `265/265`
- sign equality to stage `265/265`
- runtime errors 0, nonfinite values 0

Known dominant-color counts were `37..169`; runner-up counts were `5..26`.

## Actual-ONNX fresh audit

The two support190 seeds were rerun through the actual k33 and staged ONNX models under
ORT_DISABLE_ALL with one thread. This confirms support190's histogram prediction rather
than merely reusing it.

| seed | candidate | accuracy | staged | raw/sign equal | runtime | nonfinite | dominant count | runner-up | k33 selected channels |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 192800661 | 4998/5000 | 99.96% | 5000/5000 | 4998/5000 | 0 | 0 | 28..238 | 2..33 | 0..1 |
| 192930007 | 4997/5000 | 99.94% | 5000/5000 | 4997/5000 | 0 | 0 | 27..220 | 2..33 | 0..1 |

Aggregate: `9995/10000 = 99.95%`; raw and sign equality to stage are also
`9995/10000`. No candidate value occurs in `(0, 0.25)`.

Both seeds independently exceed the required 90%, with runtime/nonfinite 0. Therefore
they satisfy POLICY90 despite five disclosed semantic failures. All five are
false-negatives where the true dominant box color count is at most 33, so k33 selects no
channel. support190 predicted exactly 5/10000 failures, matching the actual ONNX result.

## All-support counterexamples

Actual candidate and staged models were also executed on two reachable generator inputs:

1. False-negative: three `3×3` boxes with one overwrite give box count 26 and distractor
   count 1. ArgMax selects the box color; k33 selects no color. Candidate differs in 54
   decoded cells.
2. False-positive: box count 48 and an isolated distractor count 37. ArgMax selects only
   the box color; k33 selects both channels. Candidate differs in 159 decoded cells.

The staged model is correct on both, while k33 is wrong with runtime/nonfinite 0. Thus
`all_support_exact=false` is proven by concrete executions. Under the current explicit
policy this is disclosure, not an automatic rejection: the admission criterion is each
independent normal-candidate stream being at least 90% with no execution instability.

## Integrity and action

- candidate SHA-256: `e6515b2ddf32c2eb80581aa3267e24683d2aa53d9445483b2a2a0752f94072d5`
- staged SHA-256: `19fbdce89a5c89f5ff376b2fbbdb630ead5535d5ed5ebe7d9914a4de89e5023c`
- authority/root ZIP SHA-256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `all_scores.csv` SHA-256: `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`
- all hashes match before/after
- stage action: `NONE_DO_NOT_TOUCH_STAGE`

Machine-readable evidence: `result.json`. Reproduction script:
`audit_k33_policy90.py`.
