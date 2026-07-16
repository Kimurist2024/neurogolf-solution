# Low36 target expansion — 8-task audit

## Outcome

The requested eight members were independently audited against
`submission_base_8005.16.zip` (SHA-256
`73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`).
**No safe strictly-cheaper candidate exists in the searched exact, history,
sparse-representation, and SOUND true-rule families.** This lane contributes
**+0.0**, emits no candidate, and does not build or modify any submission ZIP.

- completed: **8/8**;
- accepted: **0**;
- all eight members are byte-identical to the already-audited 8004.50 base;
- the Sakana transform reproduces every known pair: task176 25/25, tasks040,
  252, and390 266/266, and tasks127,147,149,272 267/267;
- every graph passes full checker and strict/data-propagating shape inference,
  has standard domains, and has Conv-bias UB0 (none uses Conv);
- protected ZIPs, score files, CSVs, and shared models were not modified.

## Per-task decision

| task | actual cost | strongest lower/tie lead | decision |
|---:|---:|---|---|
| 149 | 84 | cost-84 / 19–21-input Einsum lineage | **REJECT**: no below-84 history; compact graphs are giant Einsum |
| 390 | 84 | 36/56-input giant alternates | **REJECT**: no structurally safe below-84 graph; conventional floor 2699 |
| 272 | 83 | archive static-cost82 | **REJECT**: candidate cannot create an ORT session because `Max` mixes int64 and int32; incumbent has 35 runtime-shape contradictions |
| 147 | 82 | history floors 84/85/532 | **REJECT**: no strict decrease; incumbent has 33 runtime-shape contradictions |
| 040 | 80 | exact sparse-H expected param cost30 | **REJECT**: full checker reports Einsum sparse operand rank0; dense rank-2 floor is 80 |
| 176 | 80 | exact sparse-E expected param cost63 | **REJECT**: same mandatory full-check failure; only dense alternate costs210 |
| 252 | 80 | exact sparse-W expected param cost54 | **REJECT**: same mandatory full-check failure; history costs84/96 |
| 127 | 84 | 46/55/56-input Einsum lineage | **REJECT**: no below-84 history; compact graphs are giant Einsum |

The inherited giant-Einsum and shape-cloaked members are fixed LB-white
authorities, not templates that authorize a new unsafe replacement.

## Sparse exact-rewrite disposition

Tasks040, 176, and252 initially looked promising because their current graphs
are runtime-shape truthful and stay below the lane's giant-Einsum threshold.
For each, the zero-heavy initializer was converted exactly to a
`SparseTensorProto`; dense reconstruction is bit-identical. However, ONNX full
checking fails before scoring or runtime: shape inference exposes that sparse
Einsum input as rank 0 while the equation requires rank 2. Adding a dense
`value_info`/reshape would materialize and count the full tensor, making the
candidate cost-dominated. Therefore these are not official-validator-compatible
candidates, have no acceptable candidate SHA/cost, and correctly fail closed.

## Search evidence

1. The all-400 archive inventory covers 1,196 ZIPs, 448,568 ZIP members,
   233,751 loose observations, and 13,591 unique non-baseline graphs. Its sole
   relevant below-baseline lead is invalid task272 cost82.
2. The focused harvest covers another 1,134 unique different graphs. Across
   these targets it finds only ties, cost-dominated graphs, shape-cloaked
   families, or giant-Einsum alternatives.
3. Complete exact Wave2 scanned all 400 members and accepted zero, with no hit
   among these targets.
4. Known dual ORT was rerun for all eight incumbents. Tasks040,252,390 pass
   266/266, task176 passes 25/25, and tasks127/149 pass 267/267 in both modes.
   The inherited shape-cloaked tasks147/272 pass ORT_DISABLE_ALL 267/267 but
   fail default session creation. No replacement survived far enough to invoke
   the two-seed fresh gate.

Machine-readable authority: `baseline_audit.json`, `true_rule_audit.json`,
`known_dual.json`, `history_audit.json`, `sparse_build_manifest.json`,
`result.json`, and `winner_manifest.json`.
