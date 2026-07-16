# Low37 target expansion — 8-task audit

## Outcome

The requested eight members were independently audited against the immutable
`submission_base_8005.16.zip`. **No safe strictly-cheaper candidate exists in
the searched exact, archive, factorization, and decoded-rule families.** This
lane contributes **+0.0**, emits no candidate, and does not build or modify a
submission ZIP.

- baseline SHA-256:
  `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`;
- completed: **8/8**;
- accepted: **0**;
- all eight latest members are byte-identical to the already-audited 8004.50
  baseline members;
- every decoded Sakana true rule reproduces every known pair: task320 266/266,
  task154 266/266, task393 265/265, task290 266/266, task336 31/31,
  task003 265/265, task058 22/22, and task072 268/268;
- protected ZIPs, score ledgers, CSVs, and root artifacts were not changed.

## Per-task decision

| task | latest actual cost | current structure | strongest lower/tie lead | decision |
|---:|---:|---|---|---|
| 320 | 80 | one output-only 14-operand Einsum, truthful runtime shapes | no lower graph; conventional history floor 770 | **REJECT**: no strict decrease in the complete archive or exact scans |
| 154 | 88 | one prohibited 54-operand giant Einsum, truthful runtime shapes | 44/48-operand giant alternates; conventional floor 2679 | **REJECT**: compact lineages are giant and clean lineage is dominated |
| 393 | 86 | 10 nodes, three declared/runtime shape contradictions | clean/alternate floors 95, 117, 121 | **REJECT**: no lower graph; default ORT also rejects inherited TopK shape metadata |
| 290 | 91 | 9 nodes, five declared/runtime shape contradictions | static costs 73/75/88 | **REJECT**: actual profiling gives 91/93/97, so the best lead only ties |
| 336 | 92 | one prohibited 42-operand giant Einsum, truthful runtime shapes | conventional generator-compiled model cost 4746 | **REJECT**: no safe below-92 lineage |
| 003 | 78 | 5 nodes, two declared/runtime shape contradictions | same-cost 78 alternate; conventional floor 260 | **REJECT**: tie only; truthful implementation exceeds the floor |
| 058 | 78 | one output-only 16-operand Einsum, truthful runtime shapes | historical floors 86, 94, 213 | **REJECT**: no strictly-lower history member |
| 072 | 78 | 5 nodes, two declared/runtime shape contradictions | same-cost 78 alternate; other floors 120, 180 | **REJECT**: tie only; truthful XOR implementation exceeds the floor |

The current compact/cloaked/giant members are fixed LB-white authorities, not
templates that authorize a new unsafe graph. A new proposal must itself be
truthful and satisfy every current structural gate.

## Search evidence

1. The all-400 archive inventory covers **1,196 ZIPs, 448,568 ZIP members,
   233,751 loose observations, and 13,591 unique non-baseline graphs**. Among
   these targets its only retained numeric lower leads are task290's static
   73/75/88 models.
2. Runtime scoring disproves those apparent task290 savings: their actual costs
   are **91/93/97** against latest cost 91. The cost-73 graph is known/fresh
   correct but is not strictly cheaper and inherits shape-cloak behavior.
3. The focused 1,134-unique-model harvest finds no other below-latest graph.
   For task003 and task072 it finds ties only; task058 alternatives start at 86;
   task320/393 and the giant tasks have only dominated or forbidden lineages.
4. Complete exact-Wave2 and subsequent no-op, optional-default, initializer,
   fold-shape, and axes scans accept no target. Task393 cannot be downgraded to
   opset 17 because `GroupNormalization` has no such schema, and its two-value
   axes initializer is also reused to generate TopK `k=3`.
5. The decoded generator rules show why the tiny parameter floors are not
   ordinary clean-net floors: per-column half-height recoloring (320),
   reflection into a gripper box (154), global population ranking (393),
   crop/color swap (290), oriented container flood/escape (336), conditional
   stencil continuation (003), size-dependent spiral synthesis (058), and
   top/bottom XOR (072).

## Gate disposition

No proposal clears both prerequisite gates: a strictly lower **actual** cost
and safe structure. Therefore candidate known-dual and independent two-seed
fresh-dual runs were not started. Fresh testing cannot make a tie, a dominated
graph, a shape cloak, or a giant contraction adoptable.

For provenance only, the immutable incumbents were rerun on all known examples
in both ORT modes. Seven pass completely in both modes. Task393 passes 265/265
under `ORT_DISABLE_ALL` but default session creation fails because inherited
false shape metadata makes TopK appear to have fewer than three axis elements;
this further prevents treating its structure as a safe new-candidate template.

Authoritative machine-readable evidence:

- `baseline_audit.json` — latest payload hashes, actual costs, full checker,
  strict/data-prop, runtime shape traces, domains, op/fan-in inventory, and
  Conv-bias audit;
- `known_baseline_dual.json` — dual-ORT all-known provenance runs;
- `true_rule_audit.json` — decoded rule summaries and complete known-pair
  reproduction;
- `history_audit.json` — exhaustive archive, focused harvest, runtime lower-lead
  screens, and exact-pass evidence;
- `result.json` — all eight final decisions;
- `winner_manifest.json` — empty authoritative promotion list.
