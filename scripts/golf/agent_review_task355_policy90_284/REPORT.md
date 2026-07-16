# task355 cost249 independent POLICY90 audit

## Decision

**PASS — `PASS_POLICY90_INDEPENDENT_REVIEW`.** The pinned cost-249 candidate passes the requested fail-closed normal POLICY90 audit.

This is explicitly **not a claim of complete correctness**. The candidate is 264/267 on the known corpus and its official profile has `correct=false`. Task355 is absent from the repository private-zero catalog, but it is named in the public overfit-risk top-10 list; the acceptance here is therefore based on the normal 90% policy plus two independent 10,000-case fresh stress sets, not on exact correctness.

## Pinned artifacts and cost

- Candidate: `scripts/golf/loop_7999_13/lane_archive_all400/task355_r04_static249.onnx`
  - SHA-256: `7ca617858a19310a433010e6e50da46b4d562d76f3d0688665c8387bdf6f24d8`
  - file size: 2,162 bytes
  - actual and official profile: memory 227 + params 22 = **cost 249**
  - official score: 19.482547103535293; `correct=false`
- Authority: `submission_base_8009.46.zip` / `task355.onnx`
  - ZIP SHA-256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
  - member SHA-256: `f87a253a7925f7bd848135471b29dea8c0b5c3c2335162f2d7f38ea566bc8a6c`
  - actual and official profile: memory 228 + params 22 = **cost 250**
  - official score: 19.478539082137754; `correct=true`

The candidate is one cost unit lower than the exact authority, for a score gain of 0.004008021397538868. No candidate promotion or submission mutation was performed.

## Sanitized runtime results

The candidate was passed through the repository's official sanitizer before all independent runtime sessions. The sanitized graph SHA-256 is `ef91c45f2d4669f1064e22155b8985e4f7dd6512d4eb5564b8dc2751f09cbe21`.

Known data comprises all 267 cases from `inputs/neurogolf-2026/task355.json` (train 4, test 1, arc-gen 262). Fresh cases were generated directly from `task_de1cd16c.py`; each seed produced 10,000 unique inputs. Every result was identical across the four requested ORT configurations:

| ORT configuration | Known 267 | Seed 284355001 | Seed 284455001 |
|---|---:|---:|---:|
| disabled, 1 thread | 264/267 (98.8764%) | 9871/10000 (98.71%) | 9860/10000 (98.60%) |
| default, 1 thread | 264/267 (98.8764%) | 9871/10000 (98.71%) | 9860/10000 (98.60%) |
| disabled, 4 threads | 264/267 (98.8764%) | 9871/10000 (98.71%) | 9860/10000 (98.60%) |
| default, 4 threads | 264/267 (98.8764%) | 9871/10000 (98.71%) | 9860/10000 (98.60%) |

Across 12 dataset/config rows and 81,068 case/config executions:

- runtime errors: 0
- nonfinite cases/elements: 0 / 0
- output-shape mismatches: 0; observed shape only `[1,10,30,30]`
- positive values in `(0,0.25)`: 0
- sign mismatches across configurations: 0 cases / 0 cells
- raw-output mismatches across configurations: 0 cases
- minimum positive output: 1.0
- maximum nonpositive output: 0.0

The bool output therefore has a clean sign margin. The independent converter and repository converter agreed exactly on all 20,267 logical cases.

Fresh corpus identities:

- seed `284355001`: case-data SHA-256 `94fe39dd66dcc27f9ec312939df9f024a1a13c55356398de689adf0d18411142`
- seed `284455001`: case-data SHA-256 `57f18ed32828b4af47721408b0bfe36a36b21ee6f7599f902bbd1bfd552b8f62`

## Structural, lookup, giant, and cloak audit

Full ONNX checking and strict shape inference with data propagation passed. The graph has 39 nodes and 40 node outputs, with canonical float32 input and bool output shapes `[1,10,30,30]`. All inferred values are static; scalar tensors are treated as valid rank-zero static tensors.

The largest declared intermediate has 10 elements, all initializers total 22 elements, maximum node arity is 8, and there are no giant initializers, giant intermediates, nested graphs, functions, sparse/external data, banned ops, nonstandard domains, or Conv/QLinearConv bias UB findings.

The two `GatherElements` nodes were inspected rather than rejected by name:

- one selects four values from a dynamic 10-element tensor;
- one selects one value from a dynamic four-element tensor;
- neither has a giant initializer ancestor or a suspicious table/output shape.

For the cloak check, the sanitized graph exposed and executed all 40 node outputs on a known input under disabled/default optimization and 1/4 threads. Every runtime shape matched its strictly inferred shape, all intermediate values were finite, and there were no giant runtime intermediates in any configuration.

## Policy and reproduction

The private-zero catalog check found no task355 entry. The public overfit-risk source contains task355, so this report deliberately records a normal POLICY90 acceptance rather than exact correctness. All machine-readable gates in `evidence.json` are true.

Reproduce from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/golf/agent_review_task355_policy90_284/audit.py
```

The lane-283 screen was not used as audit input. Kimi was not used. This audit wrote only its lane artifacts and did not modify root or `others/71407`.
