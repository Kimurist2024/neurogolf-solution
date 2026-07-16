# task012 h8w8 POLICY90 independent review (lane 273)

## Decision

**PASS_POLICY90_INDEPENDENT_REVIEW**.

The candidate satisfies every requested 90% policy, structural, runtime-safety,
symmetry, and independence gate. This is not a claim of exact official
correctness: the candidate is 252/265 on the complete known corpus, so the
official verifier correctly reports `correct=false`. The independently
profiled cost is nevertheless 650, versus 710 for the immutable exact
authority.

No root submission, `71407`, or promotion target was written by this review.

## Immutable inputs

| Artifact | SHA-256 | Official result |
|---|---|---|
| `submission_base_8009.46.zip` | `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927` | immutable container |
| authority `task012.onnx` | `478a310e10fcf0a3e82df943fd6ab43671c47059f8e6eb675bf0004bef576500` | memory 0, params/cost 710, `correct=true` |
| reviewed candidate | `9aea31a6c01f7af21d893f6e5dde16dc947cdb17088686654f3f568845fbb947` | memory 0, params/cost 650, `correct=false` |

The ZIP member is byte-identical to `artifacts/handcrafted/task012.onnx`.
Both official measurements were rerun through `score_and_verify`; the
candidate used `require_correct=false` solely so the profiler could report its
actual cost despite its intentional POLICY90, rather than exact, semantics.
The candidate saves 60 parameters/cost units.

The audit consumes no lane272 MILP result or evidence value. Its only lane272
input is the candidate ONNX binary. Known data, generator cases, authority,
structure, scores, and runtime results are regenerated independently.

## Generator domain and color symmetry

From `task_0962bcdd.generate`, each of two columns is independently in `3..9`
and gravity is in `0..3`. The audit regenerated every parameter tuple in the
fixed order

`(col0, col1, gravity) for col0=3..9, col1=3..9, gravity=0..3`,

giving exactly `7 × 7 × 4 = 196` latent states. Two parameter pairs collapse
to duplicate grids, so these 196 tuples contain 194 unique input/output pairs;
no tuple was skipped. The state-tuple SHA-256 is
`7a81d461d40dc0dba8f05ef6f76fd32ac8ec25c80c8df8686a8f1487a7c4ad2b`.

The representative color pair was `[1,2]`. Generalization to every ordered
pair of distinct nonzero colors is proved from the serialized model itself:

- the sole node is `Conv(group=10)` with weight `[10,1,8,8]` and bias `[10]`;
- channels 1 through 9 have raw-identical 64-float weight slices, each with
  SHA-256 `2afc9bced52cf2c3ee12959c46bf658bda39466b26a10a1bc700b831d5596d73`;
- channels 1 through 9 have raw-identical float32 bias bytes `ede40fc2`;
- background channel 0 remains fixed and is allowed to have its separate
  classifier.

Thus every nonzero output channel applies the exact same depthwise classifier
to its corresponding input color channel. Relabeling colors 1..9 merely
relabels outputs, which proves nonzero-color permutation equivariance without
depending on MILP claims.

The candidate is correct on 186/196 states = **94.897959%** in all four runtime
configurations. The ten failing tuples are:

`(3,5,1), (4,6,1), (5,3,3), (5,7,1), (6,4,3), (6,8,1), (7,5,3), (7,9,1), (8,6,3), (9,7,3)`.

## Runtime results

The same generated cases were run in four independent ORT sessions:

| Configuration | Known 265 | Latent 196 | Fresh 273012001 | Fresh 273112001 |
|---|---:|---:|---:|---:|
| DISABLE_ALL, threads 1 | 252 (95.094%) | 186 (94.898%) | 9502 (95.02%) | 9472 (94.72%) |
| default optimization, threads 1 | 252 (95.094%) | 186 (94.898%) | 9502 (95.02%) | 9472 (94.72%) |
| DISABLE_ALL, threads 4 | 252 (95.094%) | 186 (94.898%) | 9502 (95.02%) | 9472 (94.72%) |
| default optimization, threads 4 | 252 (95.094%) | 186 (94.898%) | 9502 (95.02%) | 9472 (94.72%) |

Fresh generation was performed once per specified seed and the identical
case set was replayed across all four configurations:

- seed `273012001`: 10,000 cases, 7,147 unique pairs, case SHA-256
  `5e28c272eeeba774a7c6c33031d7fe46919a5a47637f93fe0175eae2f98e48e3`;
- seed `273112001`: 10,000 cases, 7,102 unique pairs, case SHA-256
  `5e0cb59e77612c6fed6e7c025d645c9bc340ddd9d5c8564d9f433766af00f59f`.

The compact one-hot converter was byte-compared with the official converter
on all 265 known, all 196 latent, and all 20,000 fresh cases: 20,461/20,461
exact, zero mismatches.

Across 16 dataset/config evaluations and 81,844 inference executions:

- runtime errors: 0;
- nonfinite cases/elements: 0 / 0;
- output-shape mismatches: 0;
- prediction-sign mismatching cases/cells versus DISABLE_ALL threads 1: 0 / 0;
- per-dataset raw-output SHA-256 also matches across all four configurations;
- outputs in the unstable interval `(0, 0.25)`: 0 elements;
- minimum positive output: `0.39035797119140625`;
- maximum nonpositive output: `-0.8662528991699219`.

The sign result is therefore stable across optimization and thread settings,
with a measured margin on both sides of the zero threshold.

## Structural and policy audit

The candidate passes:

- ONNX `full_check`;
- strict shape inference with `data_prop=True`;
- canonical static float32 input/output `[1,10,30,30]`;
- one node total: output-only `Conv` in standard domain, opset 13;
- no counted intermediate activation, nested graph, function, sparse/external
  initializer, banned op, or nonstandard domain;
- exactly 650 finite float32 parameters (`w[10,1,8,8]`, `b[10]`);
- Conv bias UB0: inferred output channels 10 and bias length 10;
- runtime output shape `[1,10,30,30]` on every execution, so no shape cloak;
- no lookup or fixture correction: the complete graph is one Conv and its
  only constants are that Conv's dense weight and bias.

Every fail-closed gate in `evidence.json` is true.

## Reproduction

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/golf/agent_review_task012_h8w8_policy90_273/audit.py
```

Artifacts are `audit.py`, `evidence.json`, and this report. No model was copied,
rewritten, or promoted by the independent review lane.
