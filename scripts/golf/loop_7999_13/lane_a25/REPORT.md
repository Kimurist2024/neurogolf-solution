# A25 — task117 / task160 SOUND rebuild audit

## Result

- Exact base: `submission_base_7999.13.zip`
- Base SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Accepted replacement: none
- Score delta: **+0.000000**
- Root ZIP / CSV / best / ledger / `artifacts/handcrafted`: untouched

No strictly cheaper model survived known correctness and truthful runtime-shape
gates.  The generator-rule controls are structurally safe but materially more
expensive than the exact 7999.13 members.

## Exact baselines

| Task | SHA-256 | Actual cost | Known audit | Runtime-shape result |
|---|---|---:|---|---|
| 117 | `e8dee03b3c5f0dc39fe1333b9b6cd53c4903aa2015baa5db15ef4a7897ac6073` | 606 | disabled 265/265; default session error | 10 declared/actual mismatches; terminal ScatterND cloak |
| 160 | `6300f4550400fc63391ee490cbb8635f468e571296dc84085c24e7aba85b8548` | 404 | 265/265, errors 0 in both | 2 mismatches at `hid`/`low` |

These baseline mechanisms were not treated as permission to introduce a new
shape-cloaked candidate.

## Generator rules

### task117 / `task_4c5c2cf0.py`

The input contains one five-cell X body and one different-color connected leg
sprite in a single quadrant.  The output reflects the leg sprite across both
axes around the X center.  The truthful control is the previously derived
count/body-center/histogram implementation:

- `task117_truthful_copy_hist`: cost 6762
- both ORT modes: 265/265, errors 0
- declared/runtime mismatches: 0
- standard domain, no giant Einsum

The compact moment/ScatterND baseline is generator-derived, but its low score
depends on a terminal `[1,1,1,1]` declaration while runtime output is
`[1,10,30,30]`.  The apparent static-278 `col_once_clean` history is
unscorable and has 95 runtime-shape mismatches.  A truthful float terminal
paint requires at least 720 update bytes before rule logic, so it cannot beat
606 through that architecture.

### task160 / `task_6c434453.py`

Five separated blue sprites are placed on a 10x10 grid.  Every hollow 3x3 box
is replaced by a red three-row plus; all other sprites remain blue.  The
truthful convolutional rule control gives:

- `task160_truthful_rule_v1`: cost 2978
- both ORT modes: 265/265, errors 0
- declared/runtime mismatches: 0
- standard domain, no giant Einsum

## Complete below-base frontier

| Candidate | Actual cost | Both-ORT known | Shape mismatches | Verdict |
|---|---:|---|---:|---|
| task160 2x1 upper renderer | 384 | 0/265 in both | 2 | reject |
| task160 2x1 lower renderer | 384 | 0/265 in both | 2 | reject |
| task160 no feature bias | 402 | 0/265 in both | 2 | reject |
| task117 col-once cloak | unscorable | 265/265 in both | 95 | reject |

task117 full-history coverage contains 81 unique models: 63 static-floor
rejects, 16 structural/static rejects, one base-identical member, and the one
unscorable lower-static model above.  task160's retained below-base inventory
contains exactly the three candidates above.

## Exact factor / parameter reuse

- task117: all 67 nodes and all 19 initializers are live; no unused initializer
  and no duplicate full initializer tensor exists.
- task160: all 6 nodes and all 7 initializers are live; no unused initializer
  and no duplicate full initializer tensor exists.
- Removing task160's two-element first-stage bias is the cost-402 history and
  destroys all 265 known outputs.
- Shrinking the 60-parameter 3x1 renderer to either 2x1 alignment gives cost
  384, but each loses one required red-plus row and fails all known cases.

## Fresh and external gates

There were zero admissible, strictly cheaper pre-fresh finalists.  Therefore
dual independent fresh5000 and external validation were not started.  Those
gates remain mandatory for any future candidate that first passes both-known,
actual-cost, strict-shape, standard-domain, no-lookup, no-UB, and no-giant-
Einsum checks.  No processing-error candidate was accepted.

Machine evidence:

- `model_manifest.json`: exact hashes and copied models
- `full_history_inventory.json`: task117 81-model coverage and task160 frontier
- `audit_rows.json`: both ORT, actual cost, and runtime-shape traces
- `factor_audit.json`: node/initializer liveness and duplicate audit
- `fresh_dual_5000.json`: pending count 0
- `external_validation_summary.json`: pending count 0
- `winner_manifest.json`: no winner, delta 0
