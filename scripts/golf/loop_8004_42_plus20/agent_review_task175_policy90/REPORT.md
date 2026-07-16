# task175 cost-145 candidate — independent POLICY90 review

## Verdict

**`ADMIT_POLICY90`**, classified as **normal POLICY90, not known-exact**.

The candidate is structurally valid, costs `145` versus authority `166`, and
scores `262/266` known in every ORT configuration. Two repository-disjoint
fresh streams reached `2000/2000` apiece in all four configurations with zero
runtime error, nonfinite value, near-zero positive, shape mismatch, or
determinism mismatch. Nothing was staged or merged.

This is deliberately not labeled SOUND or exact: four fixed known fixtures
fail, and candidate raw output is authority-identical on `0/266` known cases.

## Integrity and cost

| item | SHA-256 | memory | params | cost |
|---|---|---:|---:|---:|
| `submission_base_8009.46.zip::task175.onnx` | `0979ba8969cdfd796f0c4e0c40c1ebf062d28093ab8866801bad9f504d537945` | 0 | 166 | 166 |
| `root_sweep29/prune_latent/task175_r001.onnx` | `40a9405880836a60f100e0072b476e4383c12c7ee053eb12ada1f049ee2e8d7c` | 0 | 145 | 145 |

The authority archive itself matches the required SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
Competition-equivalent profiling reproduced both costs. The exact task-score
gain is

```text
ln(166 / 145) = 0.1352540459359689
```

## Structural, cost, and UB gates

Both models pass full ONNX checking and strict shape inference with
`data_prop=True`. Each is a standard-domain opset-12 graph with one
standard `Einsum` node and canonical float `[1,10,30,30]` input/output.
There are no intermediate node outputs: the only node output is the declared,
inferred, and runtime-confirmed `[1,10,30,30]` graph output, so memory is zero
and there is no hidden intermediate shape to cloak.

The candidate has:

- no banned op, nested graph, function, sequence, sparse initializer, external
  data, custom domain, or unresolved/dynamic output;
- no Gather/Scatter/ArgMax/TopK/TfIdf/Hardmax/Resize/CenterCropPad/AffineGrid
  lookup or cloak operator;
- no Conv-family node, and therefore no Conv bias-length UB; the general
  Conv-family checker also returns no finding;
- finite initializer values and finite runtime values in all evaluations;
- maximum initializer size 40 elements, not an input/output fixture table.

The 18-input Einsum is high-arity and is explicitly disclosed. It is inherited
unchanged from the LB-white authority rather than introduced as a new lookup:
the equation, node, I/O, and every non-initializer model field are byte-equal.
Only the shared latent `L` sizes in `C0` and `G1` change:

| initializer | authority | candidate | exact relation |
|---|---:|---:|---|
| `C0` | `[2,3,3]` | `[1,3,3]` | candidate `[0]` equals authority `[1]` |
| `G1` | `[2,4,3]` | `[1,4,3]` | candidate `[0]` equals authority `[1]` |

`Q`, `S`, `G2`, and `K` are byte-identical. In zero-based notation the
candidate retains authority `L=1` and drops authority `L=0` (the first,
one-based `L=1` component). The dropped slices contain 4 and 12 nonzero values,
respectively, so this is not an algebraic zero deletion. It removes exactly
`9 + 12 = 21` parameters.

## Known data in four ORT configurations

Configurations were optimizations disabled/enabled crossed with 1/4 intra- and
inter-op threads. Results were identical in all four:

| model | right | wrong | runtime / nonfinite / near-positive / shape | raw stability |
|---|---:|---:|---:|---|
| authority | 266 | 0 | `0 / 0 / 0 / 0` | bit-identical across configs |
| candidate | 262 | 4 | `0 / 0 / 0 / 0` | bit-identical across configs and repeat sessions |

Candidate threshold masks equal authority on `262/266`; candidate raw tensors
equal authority on `0/266`. Candidate minimum positive raw value is
`93.67971801757812`, with no value in `(0,0.25)`.

### Exact explanation of the four failures

Every wrong spatial cell is zero in the input and its transposed cell is also
zero. The candidate predicts the fixed fixture's top-left color, while the
full quotient rule requires the listed expected color.

| fixture | expected -> candidate | wrong cells | channel diffs | coordinates `(r,c)`, zero-based |
|---|---:|---:|---:|---|
| `train[0]` | `6 -> 1` | 10 (5 symmetric pairs) | 20 | `(8,9) (9,8) (11,12) (12,11) (12,13) (13,12) (12,14) (14,12) (13,14) (14,13)` |
| `train[1]` | `5 -> 6` | 6 (3 pairs) | 12 | `(3,4) (4,3) (3,5) (5,3) (4,5) (5,4)` |
| `train[2]` | `4 -> 5` | 4 (2 pairs) | 8 | `(11,12) (12,11) (13,14) (14,13)` |
| `test[0]` | `3 -> 4` | 10 (5 pairs) | 20 | `(15,16) (16,15) (15,18) (18,15) (15,19) (19,15) (16,18) (18,16) (16,19) (19,16)` |

Thus the 20/12/8/20 channel-difference counts are exactly two channels per
wrong cell: the expected channel turns off and the top-left-color channel turns
on. The decoded `inputs/sakana-gcg-2025/raw/task175.py` rule matches all
`266/266`, confirming that these are candidate failures rather than bad gold.

## Generator identifiability and private-zero classification

Task175 is absent from `docs/golf/private_zero_tasks.md`'s 51-task catalog.
More importantly, its random generator support is input-identifiable:

1. The true quotient-color field is symmetric under `(row,col)` exchange.
2. Random `generate()` retries whenever any off-diagonal symmetric pair was
   erased on both sides.
3. Therefore an erased off-diagonal cell's true color remains visible at its
   transpose.
4. Every true diagonal color equals `input[0][0]`. A legal random cutout cannot
   erase `(0,0)`: the minimum `2x2` rectangle would also erase `(0,1)` and
   `(1,0)`, causing the retry condition.

Consequently a deterministic input-only support reference preserves nonzero
cells, fills off-diagonal zeros from the transpose, and fills diagonal zeros
from `input[0][0]`. On the two new streams, this reference and the independent
raw rule each match `4000/4000`; generator-support invariant violations are
zero. The sample includes 2,601 erased diagonal cells, so this certificate is
not vacuous.

The four fixed validate fixtures call the parameterized generator path and
ignore the random retry result. They alone violate the symmetric-erasure
invariant, and they are exactly the four candidate failures. This explains the
gap without making task175 a latent/non-identifiable task. The candidate is
still labeled POLICY90 because the ONNX graph was sampled, not formally proven
equal to the support reference for every legal parameter combination.

## Independent fresh audit

Seeds `917500031` and `917500087` do not appear in repository seed records and
are disjoint from the earlier task175 streams `175224240376` and
`376240224175`.

| seed | unique cases | erased cells | erased diagonal | candidate per config |
|---:|---:|---:|---:|---:|
| 917500031 | 2000 | 102,785 | 1,289 | 2000/2000 |
| 917500087 | 2000 | 101,872 | 1,312 | 2000/2000 |

For each seed, all four ORT configurations produced bit-identical raw streams.
Every case was also rerun through a separately constructed same-configuration
session. Across 16,000 primary config-case executions plus 16,000 independent
repeat-session executions:

- correctness is `100%` per seed and per configuration;
- runtime errors, nonfinite values, `(0,0.25)` values, runtime-shape
  mismatches, repeat-session mismatches, and cross-configuration raw
  mismatches are all zero;
- the minimum positive raw value is `96.21124267578125`.

## Admission rationale and limits

The candidate clears the explicit normal POLICY90 requirements: actual strict
cost reduction, structurally truthful standard ONNX, known and independent
fresh accuracy above 90%, and zero runtime/numeric/shape instability. Task175
is neither cataloged private-zero nor generator-non-identifiable. No lookup or
shape-cloak mechanism was found.

The admission must retain the label **`NORMAL_POLICY90_NOT_KNOWN_EXACT`**. It
must not be presented as raw-equivalent, complete-known, SOUND, or an
algebraically exact prune.

## Reproduction and immutability

- `audit_independent.py`: non-promoting reproducer
- `audit_evidence.json`: full machine evidence, including all failure cells and
  per-config counters
- `review.json`: compact handoff verdict

Observed immutable-root hashes after the audit are unchanged:
`submission_base_8009.46.zip = 4eb324d7...` and
`all_scores.csv = 8c99379c...`. All writes are confined to this review
directory; `others/71407`, root submissions, ledgers, staging, and merge state
were not touched.
