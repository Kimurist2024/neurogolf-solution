# task071 historical cost186 normal-POLICY90 review (lane 276)

## Decision

**REJECT_SHAPE_CLOAK_PRE_GATE**.

The pinned historical candidate is genuinely profile-cost 186 and is not the
separate giant-Einsum/default-unstable CastLike artifact. It nevertheless has
three declared-versus-runtime shape mismatches in every smoke configuration.
Normal-POLICY90 requires truthful shapes before accuracy sampling, so the
candidate was rejected immediately and known4/fresh `2 x 10000` were not run.

No root submission or `71407` artifact was modified. A byte-identical copy is
retained only under this lane's `quarantine/` directory.

## Target identity and disambiguation

The task071 `actual_lower` row in
`agent_mid20d_88/audit/actual_lower_four_config.json` identifies:

- SHA-256: `6cc540e94a37ca160273d7cb471492913943c9bf966d60012d6944b37773c68e`
- historical baseline cost: 188
- candidate cost: 186
- three source paths, all byte-identical.

This review selected `others/2/7616/task071_rebuilt_cost186.onnx` and verified
the same SHA at all three paths and after quarantine copying. The exact model
contains one `CastLike`, but its maximum `Einsum` arity is only 4, below the
repository giant threshold of 15. It has no giant initializer. Identification
is by the pinned SHA and actual-lower row, not by `CastLike` presence.

The historical authority is `submission_base_8005.17.zip` SHA
`c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`,
whose task071 member has SHA
`61798cc38df4cde5275141ce77900eb47fb61f83987a86ba7ad36fd38fb749a6`.

## Independent profiles

| Model | Runtime-profile memory | Parameters | Cost |
|---|---:|---:|---:|
| authority | 130 | 58 | 188 |
| candidate | 100 | 86 | 186 |

Thus the candidate is a real two-unit strict-lower artifact according to the
official profiling path. Cost alone does not clear the truthful-shape gate.

## Static audit

The candidate passes:

- ONNX `full_check`;
- strict inference with `data_prop=True`;
- all inferred node outputs statically positive;
- canonical input `[1,10,30,30]`;
- standard ONNX domain only, opset 21;
- no banned op, nested graph, function, sparse/external initializer;
- 86 finite initializer elements;
- no giant `Einsum` (`max inputs = 4`) and no initializer at or above 10,000 elements;
- no lookup/fixture table: the sole `Gather` reads graph input data and uses indices computed from that input;
- Conv-bias UB0 (there are no Conv-family bias findings).

It fails canonical output declaration: the graph declares
`output [1,10,30,1]`, not `[1,10,30,30]`.

## Truthful runtime-shape gate

The first known train input was run without sanitizer shape repair, with all 24
node outputs exposed, memory pattern/reuse disabled, under:

- ORT_DISABLE_ALL, threads 1;
- default optimization, threads 1;
- ORT_DISABLE_ALL, threads 4;
- default optimization, threads 4.

All four sessions created successfully, returned no runtime error or nonfinite
value, and produced the correct first-case sign output. All four also reported
the same three shape mismatches:

| Tensor | Declared/inferred | Runtime |
|---|---:|---:|
| `idx_hidden_u8` | `[1]` | `[30]` |
| `idx_i32` | `[1]` | `[30]` |
| `output` | `[1,10,30,1]` | `[1,10,30,30]` |

This distinguishes the target from the separate default-failing giant
candidate: default execution itself works here, but the pinned cost186 model
is shape-cloaked.

Failed pre-gates are exactly:

- `canonical_static_io`;
- `all_intermediate_runtime_shapes_truthful`;
- `runtime_shape_mismatches_zero`;
- `no_shape_cloak`.

## Accuracy-test disposition

The discovery row previously reported 264/265 known. This review does not rely
on that percentage and does not reinterpret it as eligibility. The requested
order is structural/truthful/default/giant first; any shape mismatch requires
immediate rejection. Therefore:

- full known265 x four configurations: not executed;
- generator fresh seeds `276071001` and `276171001`, 10,000 each: not generated or executed;
- sign-stability sampling: not executed.

Those tests cannot rehabilitate a candidate that already fails the no-cloak
gate.

## Artifacts

- `audit.py`: reproducible SHA/profile/static/all-output runtime audit;
- `evidence.json`: machine-readable evidence and gated decision;
- `quarantine/task071_sha6cc540_REJECT_SHAPE_CLOAK.onnx`: rejected evidence candidate only.

Reproduce from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/golf/root_task071_policy90_276/audit.py
```
