# A26 — task182 / task330 SOUND rebuild audit

## Result

- Exact base: `submission_base_7999.13.zip`
- Base SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Accepted replacement: none
- Score delta: **+0.000000**
- Root ZIP / CSV / best / ledger / `artifacts/handcrafted`: untouched

No strictly cheaper candidate passed both ONNX Runtime modes, known
correctness, and truthful runtime-shape validation.  In particular, task182's
cost-990 and cost-993 exact-reuse candidates were not accepted on disabled-only
correctness or prior fresh evidence: default ORT cannot create a session for
either model, and both retain 47 declared/runtime shape mismatches.

## Exact baselines

| Task | SHA-256 | Actual cost | Known audit | Runtime-shape result |
|---|---|---:|---|---|
| 182 | `eb60c542b43cab9ddee08e33e4dbef778e7199632c9dc16a4cd29ef55bad9587` | 994 | disabled 267/267; default session error | 47 mismatches |
| 330 | `06dcb5216cb441f6d760a28fa4a5b4affa678c9001504dbd8f4f80f3bbd2d5af` | 897 | disabled 266/266; default session error | 38 mismatches |

The inherited baseline mechanisms were used only as exact lineage references;
they do not authorize a new shape-cloaked replacement.

## Generator truth and truthful controls

### task182 / `task_776ffc46.py`

The 20x20 input contains five or six separated sprites selected from ten
hard-coded shapes.  Sprite 0 is colored 2 or 3 inside a gray 7x7 box, sprite 1
has the same shape, and the other sprites are blue.  The output recolors every
sprite whose shape equals the boxed source sprite to the source color.

Two independent truthful controls were retained:

- `task182_truthful_rule_r1`: actual cost 7099, both ORT modes 267/267,
  errors 0, runtime-shape mismatches 0.
- `task182_exact_truthful_shapes`: actual cost 169429, both ORT modes
  267/267, errors 0, runtime-shape mismatches 0.

Both prove a sound rule path but are more expensive than cost 994.

### task330 / `task_d2abd087.py`

The 10x10 input contains three to six separated gray connected components with
legal bounding boxes 2x4, 3x3, or 4x2 and component sizes from four through
eight.  The output colors a component red exactly when its size is six, and
blue otherwise.  `task330_truthful_component_rect` implements that rule at
actual cost 5525 and passes 266/266 with errors 0 in both ORT modes, with zero
runtime-shape mismatches.  It cannot beat the cost-897 baseline.

## Complete retained below-base frontier

| Candidate | Actual cost | Both-ORT known | Shape mismatches | Verdict |
|---|---:|---|---:|---|
| task182 r03 exact constant reuse | 990 | disabled 267/267; default session error | 47 | reject |
| task182 r04 `s2`/`s3` reuse | 993 | disabled 267/267; default session error | 47 | reject |
| task330 pair frames | 807 | 166/266 in both | 2 | reject |
| task330 pair frames anchor | 808 | 162/266 in both | 2 | reject |
| task330 pair frames mod9 | 817 | 210/266 in both | 2 | reject |

The other retained task182 histories have actual costs 1330, 1349, 1062, and
1196, despite misleading lower static inventories; all also fail default ORT
and have 31–42 runtime-shape mismatches.

## task182 archive-order and pollution policy

The task182 r01, r02, r05, and r06 occurrences come from aggregate multi-task
or reordered archives.  They are not isolated task-level evidence and were not
treated as proof of a white candidate.  The locally reconstructed r03 and r04
reuse candidates have isolated provenance, but fail the independent structural
and default-runtime gates above.  A prior disabled-ORT fresh5000 result for r03
was 5000/5000; its default-ORT session construction still failed, so the result
does not qualify it for adoption.

This lane therefore does not infer correctness from archive score movements,
member order, or known/fresh-only success.

## Full history and exact factor audit

- task182: 609 local-history rows reviewed (600 prefilter rejects, 3 baseline
  duplicates, 6 profiles), plus all six retained archive-frontier members and
  eight actual-profile records.
- task330: all three harvested/retained below-base candidates reviewed.
- task182: all 85 nodes and all 27 initializers are live; no unused initializer
  and no duplicate full initializer tensor exists.
- task330: all 44 nodes and all 10 initializers are live; no unused initializer
  and no duplicate full initializer tensor exists.
- task182's only real-cost reuse wins are the cost-990 and cost-993 models;
  both are structurally invalid under the required runtime-shape/default-ORT
  gates.
- task330's cost-807/808/817 factorizations change the legal component rule and
  fail 56–104 of 266 known cases.

## Fresh and external gates

There were zero admissible, strictly cheaper pre-fresh finalists.  Dual
independent fresh5000 and external validation were therefore not started.
Those gates remain mandatory after both-known, actual-cost, strict-shape,
standard-domain, no-lookup, no-UB, and no-giant-Einsum checks.  No
processing-error candidate was accepted.

Machine evidence:

- `model_manifest.json`: exact hashes and copied models
- `full_history_inventory.json`: full retained/history provenance
- `audit_rows.json`: both ORT, actual cost, and runtime-shape traces
- `factor_audit.json`: node/initializer liveness and exact reuse outcomes
- `fresh_dual_5000.json`: pending count 0
- `external_validation_summary.json`: pending count 0
- `winner_manifest.json`: no winner, delta 0
