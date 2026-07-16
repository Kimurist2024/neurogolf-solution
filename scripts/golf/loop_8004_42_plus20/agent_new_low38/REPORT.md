# Low38 target expansion — 8-task audit

## Outcome

The requested eight members were independently audited against
`submission_base_8005.16.zip`. **No safe strictly-cheaper candidate survives
the actual-cost, structure, and complete-known pre-gates.** This lane contributes
`+0.0`, emits no candidate, and does not build or modify a submission ZIP.

- baseline SHA-256:
  `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`;
- completed: **8/8**;
- accepted: **0**;
- all eight members are byte-identical to the already-audited 8004.50 base;
- every compact Sakana rule reproduces every known pair (task049 268/268,
  task287 267/267, task078/task007 266/266, all others 265/265);
- protected ZIPs, score files, CSVs, and shared handcrafted models were not
  modified.

## Per-task decision

| task | actual cost | incumbent observation | strongest lower/tie lead | decision |
|---:|---:|---|---|---|
| 141 | 78 | output-only 31-input Einsum, truthful runtime shape | no lower archive or exact rewrite | **REJECT**: no strict decrease; a safe diagonal-sum rebuild cannot beat the 78-parameter floor |
| 004 | 77 | 1,991-input Sum plus shape cloak/runtime trace failure | harvest floors 78/79/80/4866 | **REJECT**: every historical control is at or above cost 77; the LB-white incumbent is not a safe template |
| 254 | 76 | 28/49-input Einsums, truthful runtime shapes | cost 42/68 giant-Einsum variants | **REJECT**: cost42 differs on 412/500 external cases; cost68 still needs 20 operands; 60 safe TT attempts failed |
| 049 | 75 | 16-node compact graph with six runtime shape mismatches | static 69/70/72 archives | **REJECT**: best lower-static graph has actual cost 88; no actual strict decrease |
| 287 | 74 | output-only 86-input Einsum, truthful runtime shape | cost-30 Gather graph | **REJECT**: known 263/267, wrong on all four train pairs |
| 078 | 72 | CenterCropPad lineage, two runtime shape mismatches | harvest floors 80/82 | **REJECT**: no below-72 graph in complete archive/exact scans |
| 095 | 72 | 38-input Einsum plus two runtime shape mismatches | giant alternate / conventional floor 208 | **REJECT**: structurally forbidden or too expensive |
| 007 | 70 | 9-input output-only Einsum, truthful runtime shape | cost-68 contraction | **REJECT**: known 260/266 in both ORT modes with correct background one-hot encoding |

Existing giant-Einsum/shape-cloak incumbents are fixed LB-white authorities,
not authorization to introduce a new unsafe candidate. A replacement must
itself have truthful runtime shapes and avoid lookup, cloak, and giant-fan-in
structures.

## Search evidence

1. The all-400 archive inventory covers 1,196 ZIPs, 448,568 ZIP members,
   233,751 loose observations, and 13,591 unique non-baseline graphs. For these
   targets it retains only task254, task049, task287, and task007 lower-static
   leads; each fails a prerequisite gate.
2. The focused harvest independently screens 1,134 unique different graphs.
   It proves that task004/task078 controls are above their incumbents and that
   task049's apparent static reduction is actual cost 88, not below 75.
3. The complete exact Wave2 all-400 pass accepts zero and finds no opportunity
   on these eight members. The exact initializer-alias scan likewise builds
   zero candidates for all eight.
4. Task254's cost-42 archive graph passes the local 265 known pairs but uses a
   33-input giant contraction and differs from the exact model on 412/500
   external threshold cases. Its cost-68 variant still needs 20 inputs. The
   dedicated safe rebuild tested 60 tensor-train candidates below cost 76;
   none solved a complete known case, while exact rank analysis gives a safe
   family floor of 114.
5. Task287's cost-30 proposal is 263/267 known, failing all four train examples
   by 24–80 cells. Task007's cost-68 proposal is 260/266 in both ORT modes; its
   six failures occur when in-grid background is correctly encoded as channel
   zero, producing 16–21 extra positive cells.
6. The decoded rules confirm the global nature of the remaining transforms:
   full-diagonal summation (141), cross-row right shifts (004), rare-color
   extraction/cropping (049), fixed-column sorting (078), and two-pass local
   dilation (095). Conventional truthful graphs exceed the present 72–78
   compact floors wherever history supplies a control.

## Gate disposition

No proposal clears all three prerequisite gates: strictly lower **actual**
cost, safe structure, and known100 in both ORT modes. Therefore two independent
fresh-dual runs were not started. Fresh accuracy cannot make a same-cost,
known-false, shape-cloaked, giant-fan-in, or actual-more-expensive graph
adoptable.

Authoritative machine-readable evidence:

- `baseline_audit.json` — member hashes, actual costs, full checker,
  strict/data-prop, domains, runtime shape traces, op/fan-in inventory, and
  Conv-bias audit;
- `true_rule_audit.json` — readable rules and complete known-pair reproduction;
- `history_audit.json` — all-400 archive, focused harvest, exact scans, known
  lower-lead profiles, and task254 counterevidence;
- `evidence/task007_cost68_known_dual.json` — reproducible 260/266 result in
  default and ORT_DISABLE_ALL with correct channel-zero background encoding;
- `result.json` — all eight final decisions;
- `winner_manifest.json` — empty authoritative promotion list.
