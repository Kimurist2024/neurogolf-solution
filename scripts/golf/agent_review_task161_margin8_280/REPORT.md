# task161 margin8 independent audit

## Decision

**PASS — `PASS_MARGIN8_INDEPENDENT_REVIEW`.** The pinned cost-186 candidate passes the requested normal POLICY90 review independently of lane 279's evidence. All evidence gates are true.

This is a POLICY90 acceptance, not exact official correctness. The official scorer reports `correct=false` for the candidate because it matches 265/266 known cases, while every requested known/fresh evaluation is above 99% and therefore above the 90% policy threshold.

## Pinned identities and cost

- Candidate: `scripts/golf/root_task161_margin_repair_279/candidates/task161_cost186_margin8.onnx`
  - SHA-256: `57487cce1b40cc7df6097cdf1e82e7bfa53b9bcb6f5be954329ea10d132ced81`
  - actual profile: memory 120 + params 66 = **cost 186**
  - official profile: cost 186, score 19.774253326286797, `correct=false`
- Unscaled source: `scripts/golf/loop_7999_13/lane_archive_all400/task161_r01_static186.onnx`
  - SHA-256: `6752eeea166c8111cda053c3cc36f54b1409d81c7553d672201792f646b31e3a`
  - actual profile: memory 120 + params 66 = cost 186
- Authority: `submission_base_8009.46.zip` / `task161.onnx`
  - ZIP SHA-256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
  - member SHA-256: `5dc274d8515f1ac2a5c58583197984cd60fa2ede69fbe8206992f98940a38fbe`
  - actual and official profile: memory 120 + params 70 = **cost 190**; official `correct=true`

## Margin repair proof

The candidate and source each contain the same three serialized graph nodes. `spatial` and `bias` are byte-identical; the only changed initializer is float32 `[2,2]` `poly`. Its candidate raw bytes equal `np.float32(source_poly * 8)` exactly:

```text
source    [[ 0.0000015859797, -0.0561312698],
           [-0.0000302170865,  0.0596185289]]
candidate [[ 0.0000126878376, -0.4490501583],
           [-0.0002417366923,  0.4769482315]]
```

Restoring the source `poly` makes the entire serialized candidate byte-identical to the source. `poly` has exactly one consumer: operand 3 of the final `Einsum`, which directly produces `output`. The change therefore gives a uniform positive output scale of 8 without changing signs. Runtime testing strengthens this graph proof: on every evaluated case and all four ORT configurations, candidate raw output bytes were exactly the raw bytes of `source_output * np.float32(8)`, including signed-zero representation. There were zero scale mismatches and zero source/candidate sign mismatches.

## Runtime results

Known data was all 266 cases from `inputs/neurogolf-2026/task161.json` (train 3, test 1, arc-gen 262). Fresh cases were independently regenerated from `task_6cdd2623.py`; each seed produced 10,000 unique inputs. Results were identical across all four ORT configurations:

| ORT configuration | Known 266 | Seed 280161001 | Seed 280261001 |
|---|---:|---:|---:|
| optimizations disabled, 1 thread | 265/266 (99.6241%) | 9924/10000 (99.24%) | 9935/10000 (99.35%) |
| default optimizations, 1 thread | 265/266 (99.6241%) | 9924/10000 (99.24%) | 9935/10000 (99.35%) |
| optimizations disabled, 4 threads | 265/266 (99.6241%) | 9924/10000 (99.24%) | 9935/10000 (99.35%) |
| default optimizations, 4 threads | 265/266 (99.6241%) | 9924/10000 (99.24%) | 9935/10000 (99.35%) |

Across the 12 dataset/config evaluations there were 81,064 candidate executions and 81,064 paired source executions. Candidate/source runtime errors, nonfinite elements, and output-shape mismatches were all zero. Raw-output and sign hashes were stable across configurations. Candidate values in `(0, 0.25)` numbered zero; the global minimum positive value was 1.7959184646606445 and the maximum nonpositive value was 0.0. The reference converter was cross-checked exactly on all 20,266 distinct known/fresh cases.

Fresh case-data SHA-256 values:

- seed `280161001`: `6bc87af2d34eb367f7f7c5a04da2e2580052e4ac9af412477b7100a14e039689`
- seed `280261001`: `263af321452baf309fc76b867286dd3ebb8d5a29c9cbe31350fa07a14b00b019`

## Structural and policy gates

- ONNX full checker and strict checker with data propagation: pass.
- Standard opset 18 only; three nodes (`Einsum`, `Add`, `Einsum`).
- Canonical static float32 input/output `[1,10,30,30]`; every typed intermediate was truthfully realized under all four ORT configurations.
- No lookup ops, banned ops, nonstandard domains, nested graphs, functions, sparse/external data, giant initializer, giant Einsum, nonfinite initializer, or Conv-bias UB0 issue.
- Maximum Einsum arity is 8, below the giant threshold of 15.
- Fail-closed POLICY90 threshold 0.90: all known and fresh configuration rows pass.
- No candidate promotion was performed. `root/others/71407` was not written. Kimi was not used.

Machine-readable evidence is in `evidence.json`; the reproducible independent auditor is `audit.py` in this directory.
