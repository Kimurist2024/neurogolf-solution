# Low45 target expansion — 8-task audit

## Outcome

The additional eight files were independently audited against
`submission_base_8005.16.zip`. **No safe strictly-cheaper candidate exists in
the searched exact, complete-history, task-specific, and decoded-rule
families.** This lane contributes **+0.0**, emits no candidate, and does not
build or modify a submission ZIP.

- baseline SHA-256: `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`;
- completed: **8/8**;
- accepted: **0**;
- all eight members are byte-identical to the 8004.50 base;
- all eight incumbents pass full checker, strict/data-prop, truthful runtime
  shapes, standard-domain, Conv-bias UB0, and known100 in both ORT modes;
- every decoded Sakana rule reproduces the complete known corpus;
- protected ZIPs, score files, and CSVs were not modified.

## Per-task decision

| task | cost | current structure | known disable | known default | strongest lower/tie lead | decision |
|---:|---:|---|---:|---:|---|---|
| 024 | 30 | Einsumx1 | 266/266 | 266/266 | no graph below cost 30; focused alternate costs 56 | reject |
| 113 | 30 | Gatherx1 | 265/265 | 265/265 | cost-30 one-node Gather tie | reject |
| 385 | 30 | Gatherx1 | 265/265 | 265/265 | five apparent static-cost 0/1 archive graphs | reject |
| 389 | 30 | Einsumx1 | 266/266 | 266/266 | cost-20 seven-factor approximation | reject |
| 296 | 28 | ConvTransposex1 | 268/268 | 268/268 | factored ConvTranspose selector area 18 ties cost 28 | reject |
| 399 | 25 | Castx1, ConvIntegerx1, Einsumx1 | 273/273 | 273/273 | cost-25 same-cost history member | reject |
| 359 | 24 | Einsumx1 | 266/266 | 266/266 | no sound graph below giant-Einsum cost 24 | reject |
| 110 | 24 | Einsumx1 | 266/266 | 266/266 | only cost-24/low giant-Einsum lineages | reject |

Task359 and task110 are fixed LB-white giant-Einsum incumbents. Their lineage
does not authorize a new unsafe model: any replacement would still need to
pass the requested no-giant/truthful/dual-ORT/fresh guarantee gates. No such
strictly-cheaper replacement exists.

## Search evidence

1. The all-400 inventory covers **1,196 ZIPs, 448,568 ZIP members, 233,751
   loose observations, and 13,591 unique non-baseline graphs**. It retains only
   six numeric lower leads for these targets: five task385 artifacts and one
   task389 artifact. All six fail both gold checks and all 20 fresh cases.
2. The focused harvest finds no strict safe decrease: task024's alternate
   costs 56; task113/task385/task389 have cost-30 ties or dominated graphs;
   task296 costs 90; task399 has a cost-25 tie; task110's low artifacts remain
   18/29-input giant Einsums; and task359 has no sound lower authority.
3. The complete exact Wave2 pass produces zero opportunities or candidates for
   these eight members. The all-400 initializer-alias scan also builds zero
   candidate for them.
4. Task113's dedicated search tested 268,968 additional low-parameter
   final-output candidates, alongside earlier Pad/pool/Conv/Resize screens,
   without a sub-30 survivor.
5. Task296's calibrated factored ConvTranspose search finds no valid selector
   area <=17; area 18 only ties the current cost 28.
6. Task359's generator-exact reconstruction needs row/column histograms and
   orientation scoring. Even one natural 300-value histogram exceeds its
   fixed cost-24 giant-Einsum incumbent.
7. Task110's periodic-pattern restoration likewise has no ordinary one-node
   operator that fits below cost 24; the only compact history remains giant
   Einsum, which is forbidden for a new candidate.

## Gate disposition

No proposal clears the prerequisite strictly-lower actual-cost and safe-
structure gates. Candidate known and fresh tests therefore were not started;
fresh evaluation cannot make a same-cost, known-false, cost-dominated, or
structurally forbidden graph adoptable.

Authoritative evidence:

- `baseline_audit.json` — latest hashes, actual costs, checker/strict,
  runtime-shape, domain, op/fan-in, and Conv-bias evidence;
- `known_baseline_dual.json` — complete known corpus under both ORT modes;
- `true_rule_audit.json` — readable rule summaries and known reproduction;
- `history_audit.json` — exhaustive archive, focused harvest, exact passes,
  and task-specific proof pointers;
- `result.json` — eight final decisions;
- `winner_manifest.json` — empty authoritative promotion list.
