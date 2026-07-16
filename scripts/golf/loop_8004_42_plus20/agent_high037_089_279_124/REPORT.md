# Lane 124 — task037/task089/task279 sound memshave audit

## Outcome

No safe strictly cheaper candidate was found. Winner count is **0**, cost delta
is **0**, and projected score gain is **+0.0**. This lane did not edit
`submission.zip`, `all_scores.csv`, `others/`, or shared `artifacts/`.

The 8009.46 authority archive observed before and after the run was SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.
All three baseline snapshots are byte-identical to their authority members.

## Independent true-rule verification

`run_lane.py` implements the generator rules independently of the ONNX
graphs:

- **task037 (`1f876c06`)**: connect each same-colored pair of diagonal
  endpoints and fill the diagonal segment.
- **task089 (`3e980e27`)**: recover each visible 3x3 sprite from its complete
  copy, then stamp it at marker-only copies; red copies are horizontally
  mirrored and green copies retain orientation.
- **task279 (`b2862040`)**: a blue four-connected component is a closed box
  exactly when its component graph contains a cycle; recolor that whole
  component cyan, including attached barnacles.

Each independent solver matched every stored case and two independently seeded
fresh streams:

| task | known | fresh seed A | fresh seed B | errors |
|---:|---:|---:|---:|---:|
| 037 | 266/266 | 1500/1500 | 1500/1500 | 0 |
| 089 | 267/267 | 1500/1500 | 1500/1500 | 0 |
| 279 | 266/266 | 1500/1500 | 1500/1500 | 0 |

The exact seeds and counters are in `audit/reference_audit.json`. These 9000
fresh cases validate the reconstructed rules, not any rejected ONNX.

## Authority members

| task | member SHA-256 | official-like cost | known disabled ORT | known default ORT | truthful-runtime finding |
|---:|---|---:|---:|---:|---|
| 037 | `df9298f3b9e851bd815be5f53b8de142cbf944297477d73fc27ddc343d49c90b` | 320 = 314 + 6 | 266/266 | session-load failure | runtime planner shape mismatch; truthful trace cannot complete |
| 089 | `89183f12515ceee79eb73d69ce074a6a93145930e5fb9eb426d3a1f7f58e5607` | 1340 = 1311 + 29 | 267/267 | session-load failure | 49 declared/actual contradictions; 137,714 one-example intermediate bytes |
| 279 | `d3bb22792a3e44e09d21971f88642622a32d161f3a7888f9d0e9efe5862d0a9b` | 397 = 385 + 12 | 266/266 | 266/266 | 203 declared/actual contradictions; 615,924 one-example intermediate bytes |

All three exact members pass the repository full checker and strict shape
inference with data propagation. That static pass does not make the shape
contracts truthful: the independent runtime trace exposes the allocator
failure/contradictions above. Existing LB whiteness of the exact incumbent is
not inherited by a modified model.

## Exact mechanical scan

The all-input semantics-preserving scan covered dead nodes/initializers,
initializer deduplication, optional outputs, no-ops, CSE, constant folding,
constant absorption, combined passes, and advisory `value_info` normalization.

Seven byte-distinct variants were emitted:

- **3 `REJECT_NOT_STRICTLY_LOWER`**: metadata-only normalization on
  task037/task089/task279 did not produce a scoreable strict decrease.
- **3 `REJECT_CHECKER_OR_STRICT_SHAPE`**: task089 folding/combined variants
  violate structural or strict-shape gates.
- **1 `REJECT_OFFICIAL_NOT_CORRECT_LOWER`**: removing task089's apparently
  dead `ReduceMax` lowers the nominal cost from 1340 to 1171, but the official
  run is `correct=false`; disabled ORT records **0 right / 267 errors** and
  default ORT cannot create a session.

This reproduces the decisive task089 finding from the earlier exact-white
scan. The removed tensor was only mathematically dead; the incumbent depends
on inconsistent shape metadata and ORT buffer-planner behavior, so this is not
a portable semantic memshave.

For task037 and task279, no semantic cleanup action was available. A truthful
rebuild must materialize diagonal propagation or component topology rather
than reuse the incumbent's shape contracts; prior SOUND audits already place
those honest implementations far above the present cloaked costs. This lane
therefore rejects shape-cloak repair as non-lower instead of treating the
current charged memory as an honest optimization floor.

No ONNX reached the prerequisite intersection of strict lower official cost,
known correctness in both ORT configurations, truthful runtime shapes, full
checker, and strict data-propagating shape inference. Consequently no candidate
advanced to ONNX fresh testing or admission.

## Evidence

- `audit/results.json`: authority structures/costs, full/strict checks,
  dual-ORT known results, runtime-shape traces, all seven variants and
  fail-closed stages.
- `audit/reference_audit.json`: known and two-seed fresh verification of the
  independent executable specifications.
- `manifest.json`: authority hashes and empty winner list.
- `baseline/`: byte-exact authority member snapshots.
- `candidates/`: rejected mechanical variants only.
- `candidate/README.md`: explicit no-winner marker.
- `run_lane.py`: reproducible scanner and independent rule audit.

