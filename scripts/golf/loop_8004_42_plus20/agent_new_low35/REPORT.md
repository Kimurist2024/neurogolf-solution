# Low35 target expansion — 8-task audit

## Outcome

The requested eight members were independently audited against
`submission_base_8005.16.zip`. **No safe strictly-cheaper candidate exists in
the searched exact, history, factorization, and SOUND true-rule families.**
This lane contributes **+0.0**, emits no candidate, and does not build or modify
any submission ZIP.

- baseline SHA-256:
  `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`;
- completed: **8/8**;
- accepted: **0**;
- all eight members are byte-identical to the already-audited 8004.50 base;
- the compact Sakana true rule reproduces every known pair for every target
  (task050 271/271, task350 267/267, and 266/266 for each remaining task);
- protected ZIPs, score files, CSVs, and shared handcrafted models were not
  modified.

## Per-task decision

| task | actual cost | current structure | strongest lower/tie lead | decision |
|---:|---:|---|---|---|
| 050 | 88 | one output-only Einsum, 35 inputs, truthful runtime shape | four cost-84 transition factorizations | **REJECT**: all four fail train[0] in both ORT modes; dedicated 566-file / 18-unique history scan found no other lower graph |
| 329 | 88 | one output-only Einsum, 15 inputs, truthful runtime shape | cost-88 incumbent; conventional control floor 1050 | **REJECT**: no strict decrease in the complete archive or exact scans |
| 350 | 88 | one output-only Einsum, 33 inputs, truthful runtime shape | cost-60 rank-one approximation | **REJECT**: reconstruction is inexact and both ORT modes score 0/100 with 100 runtime/output failures |
| 356 | 88 | one output-only Einsum, 29 inputs, truthful runtime shape | cost-60 rank-one approximation | **REJECT**: reconstruction is inexact and both ORT modes score 0/100 with 100 runtime/output failures |
| 371 | 88 | one output-only Einsum, 20 inputs, truthful runtime shape | cost-88 compact alternates | **REJECT**: ties only and retains giant-Einsum lineage; conventional floor 467 |
| 360 | 86 | one output-only Einsum, 53 inputs, truthful runtime shape | cost-86 incumbent; conventional floor 340 | **REJECT**: no below-86 graph exists in the searched history |
| 214 | 85 | 10 nodes, ScatterElements, five declared/runtime shape mismatches | static cost-75 archive graph | **REJECT**: known gold false and fresh 0/20; actual cost unavailable; Identity shave also fails checker/strict inference |
| 083 | 84 | one output-only Einsum, 55 inputs, truthful runtime shape | 49-input giant-Einsum alternate; conventional floor 210 | **REJECT**: structurally forbidden and no established lower actual cost |

The existing giant-Einsum/shape-cloak members are fixed LB-white authorities,
not templates that authorize a new unsafe candidate. Under the requested policy,
a new graph must itself have truthful runtime shapes and must not introduce or
reuse giant-Einsum, lookup, or cloak structures.

## Search evidence

1. The all-400 archive inventory covers **1,196 ZIPs, 448,568 ZIP members,
   233,751 loose observations, and 13,591 unique non-baseline graphs**. For these
   eight tasks it retains only three static lower leads: task214 at 75 and
   tasks350/356 at 60. Every one fails the prerequisite correctness screen.
2. The independent focused harvest covers another 1,134 unique different
   graphs. It finds only same-cost/giant alternatives or conventional models
   above the current cost for these targets.
3. The complete exact Wave2 pass scans all 400 members and accepts zero. Its
   only hit among these targets is task214 Identity removal, which fails full
   checker and strict shape inference because the inherited `CenterCropPad`
   declaration contradicts the inferred shape.
4. The dedicated task050 factorization lane audits 566 files / 18 unique models.
   Its only strictly-lower proposals are the four cost-84 common-transition
   approximations; each is already wrong on the first train pair in both ORT
   modes.
5. The task350/task356 cost-60 rank-one proposals replace a 60-value initializer
   with a 32-value approximation whose maximum reconstruction error is 32.
   Both dual-ORT screens return 0/100, so the apparent parameter win is not an
   exact rewrite.
6. The decoded true rules confirm that these are fixed-geometry or data-dependent
   global transforms: axial between-marker filling (050/350), variable center
   selection (329), prefix/suffix row-column logic (356), midpoint localization
   (371), folding/permutation (360/214), and four-way reflection (083). A
   conventional truthful implementation starts above the 84–88 output-only
   parameter floors wherever such history exists.

## Gate disposition

No proposal clears the two prerequisite gates: strictly lower **actual** cost
and safe structure. Therefore known-dual and two independent fresh-dual runs
were not started for a promotion candidate. Fresh testing cannot make a
same-cost, known-false, non-executable, or structurally forbidden graph
adoptable.

Authoritative machine-readable evidence:

- `baseline_audit.json` — latest member hashes, actual costs, full checker,
  strict/data-prop, runtime shape traces, domains, op/fan-in inventory, and
  Conv-bias audit;
- `true_rule_audit.json` — readable rule summaries and full known-pair rule
  reproduction;
- `history_audit.json` — complete archive, focused harvest, exact Wave2,
  task050, and rank-one evidence;
- `result.json` — all eight final decisions;
- `winner_manifest.json` — empty authoritative promotion list.
