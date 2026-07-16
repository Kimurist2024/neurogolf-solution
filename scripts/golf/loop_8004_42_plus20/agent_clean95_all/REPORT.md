# Clean policy-90 all-history rescreen

## Outcome

- Baseline: `submission_base_8004.50.zip`
- Baseline SHA-256: `63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`
- Submission ZIP integration: **not performed**
- Clean policy-90 adoptees: **0**
- Projected gain: **+0.0**
- Final verdict: **NO_CLEAN_POLICY90_CANDIDATE**

The already-adopted task009 candidate
`b265f7f83d8fbf66c9388b9edfe0111d2b77a4b610377a3994a9c483fb445d28`
was explicitly ineligible for this lane.

## Inventory coverage

The two requested inventories were reused without rescanning or modifying the
repository archives:

- `agent_history_miner/history_inventory.json`: 50 accepted-history rows;
- `lane_archive_loose_sweep/inventory.json`: 441 retained rows.

After exact `(task, SHA-256)` deduplication and removal of 30 models already
equal to the 8004.50 member, 452 non-current candidates remained across 92
tasks. The other 308 baseline tasks had no candidate in either requested
inventory. Every task from 1 through 400 was therefore accounted for as either
candidate-bearing or inventory-empty. There were no unresolved source paths or
SHA mismatches.

## Gate funnel

| terminal stage | candidates |
|---|---:|
| static/policy reject | 356 |
| statically valid but not cheaper | 7 |
| actual runtime cost not cheaper / not profileable | 77 |
| known dual-ORT reject | 10 |
| fresh-500 reject (<90%) | 2 |
| independent two-seed 5000 confirmation | 0 |
| admitted | **0** |

The static gate required full ONNX checking, strict shape inference with data
propagation, positive static shapes, canonical I/O, no banned/nested/custom
constructs, no sparse/external data, no lookup/TfIdf/giant-initializer graph,
no giant Einsum, and no Conv-family bias-length UB. Cheap models from the
project private-zero/unsound-incumbent catalog were isolated at the policy
gate rather than recycled under an innocuous copied filename.

The previously observed 90--95% models for tasks 048, 365, and 396 remain
ineligible because all three tasks are in that explicit private-zero catalog;
lowering the numerical accuracy threshold does not override the clean-policy
gate. The old task396 family produced no clean 90% lead (its cheaper models are
lookup/private-zero lineage).

For completeness, the eight old 90--95% rows that were otherwise executable
were rerun on two independent 5000-case seeds. All seven task048 SHA variants
had identical predictions: 4503/5000 (90.06%) and 4554/5000 (91.08%). The
task365 SHA scored 4596/5000 (91.92%) and 4563/5000 (91.26%). Both modes agreed
and runtime errors were zero. These results numerically clear policy90, but the
hard private-zero catalog gate still rejects them, so they do not appear in the
admit manifest.

Static rejection reasons overlap when a candidate violated more than one
rule. The main counts were: private-zero catalog 205, non-static shapes 170,
giant Einsum 58, lookup/giant initializer 58, and Conv-family bias UB 33.

## Final fresh candidates

Only two models survived actual-cost, complete known dual-ORT, and runtime
shape-truth gates. Both were task023 historical models and both failed the
initial fresh-500 threshold with zero runtime errors in both ORT modes:

| task | current | candidate | gain | SHA-256 | fresh disable/default | decision |
|---:|---:|---:|---:|---|---:|---|
| 023 | 1622 | 1497 | +0.080196850 | `61313447a8f811f65257ae079330a956e63e8722daf4197f62316649e31798a7` | 1/500, 1/500 | reject |
| 023 | 1622 | 1541 | +0.051228399 | `9a2b7813889112837080e0364d0a8971671ca384079e188f460dee6b158b3ab1` | 448/500 (89.6%), 448/500 | reject |

Because no clean candidate reached 90% in the fresh-500 screen, running the two
independent 5000-case confirmations would not change an admission decision and
was correctly skipped.

## Evidence

- `inventory_union.json`: source resolution and deduplication ledger
- `screen_results.json`: every candidate's terminal stage and detailed gates
- `evidence/task023_fresh_dual_500.json`: shared independent fresh cases
- `policy90_reclassification.json`: two-seed confirmation of catalog-excluded
  90--95% rows
- `admit_manifest.json`: authoritative root handoff (empty)

The isolated files under `candidates/` are rejected evidence only and are not
merge-authoritative.
