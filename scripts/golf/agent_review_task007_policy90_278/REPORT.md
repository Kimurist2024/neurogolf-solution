# task007 cost68 independent POLICY90 review (seed lane 278)

## Decision

`PASS_POLICY90_INDEPENDENT_REVIEW`

The archived candidate is independently accepted as a normal POLICY90 candidate for
task007. This is deliberately **not** a claim of exact correctness: the candidate is
260/266 on the known corpus, while the pinned cost70 authority is exactly correct.
The candidate was audited in place and was not promoted or copied into a submission.

## Pinned artifacts and cost

- Candidate: `scripts/golf/loop_7999_13/lane_archive_all400/task007_r01_static68.onnx`
- Candidate SHA-256: `fa22f345634e3f059b0b2d334e6b9d85d60973d5cc2a6c92003b8f7cfc60486a`
- Candidate profile: memory 0, params 68, cost 68, score 20.780492294823894
- Authority archive: `submission_base_8009.46.zip`
- Authority archive SHA-256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- Authority `task007.onnx` SHA-256: `fc02d641241760fe6fa7e7ef1be2ba9aa492e7cfe42d94778ea06016573ce0b3`
- Authority profile: memory 0, params 70, cost 70, score 20.75150475795064, exact-correct
- Candidate delta: cost -2 and score +0.028987536873252187 versus the authority

Both profiles were recomputed through the repository scoring code. The candidate
identity and authority archive/member identities were fail-closed SHA checks.

## Independent accuracy audit

| Corpus | Cases | Correct | Accuracy | POLICY90 |
|---|---:|---:|---:|---:|
| Known train/test/arc-gen | 266 | 260 | 97.7444% | pass |
| Fresh seed 278007001 | 10,000 | 9,745 | 97.4500% | pass |
| Fresh seed 278107001 | 10,000 | 9,752 | 97.5200% | pass |

Each corpus was evaluated under four ONNX Runtime configurations:
`ORT_DISABLE_ALL` with 1 and 4 threads, and `ORT_ENABLE_ALL` with 1 and 4
threads. Accuracy, sign hashes, and raw output hashes were identical across all
four configurations for each corpus.

The fresh cases came directly from task007's generator with the two requested
seeds. The root seed-277 audit was not used as runtime input. An independent
one-hot input/output converter was also checked against the repository converter
on all 20,266 logical cases, with zero mismatches.

## Runtime and numerical gates

Across 81,064 case/config executions:

- execution errors: 0
- non-finite cases/elements: 0 / 0
- runtime output-shape mismatches: 0
- observed runtime shape: only `[1, 10, 30, 30]`
- positive outputs in the ambiguous `(0, 0.25)` interval: 0
- sign mismatches across ORT configurations: 0 cases / 0 cells
- minimum positive output: `2.61369452715673e+32`
- maximum non-positive output: `0`

## Structural gates

Both full ONNX checking and strict checker data propagation passed. The model has
canonical float32 input/output shapes `[1, 10, 30, 30]`; every node output is
typed, static, and positive-dimensional. The 509-byte graph contains one standard
domain `Einsum`, 68 finite initializer elements, no external data, functions,
nested graphs, sparse initializers, banned ops, lookup ops, value-info shape cloak,
giant initializer, or giant-Einsum finding. Its ten-input equation is:

`ncyx,ncuv,ncwx,ncyz,ay,bx,pr,qs,tab,tpq->ncrs`

## Reproduction

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/golf/agent_review_task007_policy90_278/audit.py
```

The machine-readable record is `evidence.json`. The audit is fail-closed: any
identity, structure, cost, accuracy, runtime, numerical, converter, or
cross-configuration stability gate failure exits nonzero. Kimi was not used, and
this lane did not write root, another agent's lane, or `others/71407`.
