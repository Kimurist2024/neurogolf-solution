# Low41 target expansion — 8-task audit

## Outcome

The requested eight members were independently audited against
`submission_base_8005.16.zip`. **No safe strictly-cheaper candidate exists in
the searched exact, full-history, and decoded true-rule families.** This lane
contributes **+0.0**, emits no candidate, and does not build or modify a
submission ZIP.

- baseline SHA-256:
  `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`;
- completed: **8/8**;
- accepted: **0**;
- all eight members are byte-identical to the already-audited 8004.50 base;
- the compact Sakana true rule reproduces every known pair for every target
  (265–269 correct, zero wrong/error per task);
- protected ZIPs, score files, CSVs, and shared models were not modified.

## Per-task decision

| task | actual cost | current structure | strongest lower/tie lead | decision |
|---:|---:|---|---|---|
| 380 | 60 | one output-only Einsum, nine inputs, truthful runtime shapes | best different history floor 99 | **REJECT**: no strict decrease in 13,591 archive graphs, 1,134 focused graphs, or exact Wave2; shrinking the 30x2 initializer breaks the contracted dimension |
| 242 | 58 | 12 nodes, seven runtime shape contradictions | cost-58 tie; other floors 64/422 | **REJECT**: no strict decrease and incumbent lineage is shape-cloaked |
| 298 | 58 | three nodes, one runtime shape contradiction | best different history floor 59 | **REJECT**: all alternatives are more expensive and the incumbent is shape-cloaked |
| 026 | 57 | QLinearConv/ConvInteger, one runtime shape contradiction | best different history floor 58 | **REJECT**: all alternatives are more expensive and no truthful exact rewrite exists |
| 261 | 57 | 16 nodes, eight runtime shape contradictions | best different history floor 59 | **REJECT**: allocator-sensitive CenterCropPad chain and no lower graph |
| 351 | 57 | 15 nodes, ten runtime shape contradictions | cost-57 tie; other floors 75/995 | **REJECT**: tie only and no truthful lower graph |
| 274 | 55 | five GatherND nodes, two runtime shape contradictions | best different history floor 64 | **REJECT**: lookup/cloak lineage is ineligible and history is more expensive |
| 317 | 55 | Bernoulli plus two Resize nodes, two runtime shape contradictions | cost-55 tie; other floor 146 | **REJECT**: tie only, nondeterministic/cloaked lineage, no lower graph |

The only structurally truthful incumbent in this lane is task380. Its fixed
3x3 clockwise rotation is already expressed as a single output-only contraction
with zero intermediate memory. The initializer contains only six nonzero values,
but cost counts its required 30x2 shape, and deleting zero rows makes the Einsum
dimension incompatible with the 30-wide input. Adding a crop or pad would
materialize a counted intermediate and cannot beat cost 60.

## Search evidence

1. The all-400 archive inventory covers **1,196 ZIPs, 448,568 ZIP members,
   233,751 loose observations, and 13,591 unique different graphs**. It retains
   no below-baseline graph for any of the eight low41 targets.
2. The independent focused harvest covers another **1,134 unique different
   graphs**. Its 19 relevant rows are all ties or strictly more expensive.
3. The complete exact Wave2 pass finds no low41 initializer alias, dead code,
   no-op node, duplicate producer, unused optional output, or annotation-only
   price reduction. Target hits: zero.
4. The decoded rules cover fixed rotation (380), row-prefix selection (242),
   cyclic color substitution (298), center-conditioned row mapping (026), row
   rotation/modulo (261), marker-relative 5x5 extraction (351), column-count
   encoding (274), and 3x3 block replication/thresholding (317). Each rule is
   perfect on every known pair, so the audit is not based on example fitting.
5. Seven incumbents rely on runtime shape contradictions, lookup, or
   nondeterminism. Their LB-white status is authority for those exact bytes only;
   it does not permit manufacturing a new unsafe candidate from the same lineage.

## Gate disposition

No proposal clears the prerequisite strictly-lower actual-cost gate. Seven
tasks also fail the safe-structure prerequisite. Therefore known-dual and two
independent fresh-dual runs were not started for a promotion candidate: such
tests cannot make a tie, a more-expensive graph, or an unsafe lineage adoptable.

Authoritative machine-readable evidence:

- `baseline_audit.json` — hashes, actual costs, full checker,
  strict/data-prop, runtime shape traces, domains, op inventory, and Conv-bias
  audit;
- `true_rule_audit.json` — readable rules and all-known-pair reproduction;
- `history_audit.json` — complete archive, focused harvest, and exact Wave2
  evidence;
- `result.json` — all eight final decisions;
- `winner_manifest.json` — empty authoritative promotion list.
