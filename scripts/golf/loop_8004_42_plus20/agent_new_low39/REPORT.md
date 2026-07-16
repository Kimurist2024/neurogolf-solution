# Low39 target expansion — 8-task audit

## Outcome

The requested eight members were independently audited against the immutable
`submission_base_8005.16.zip`. **No safe strictly-cheaper candidate survives
the actual-cost, structure, runtime-shape, and complete-known pre-gates.** This
lane contributes `+0.0`, emits no candidate, and does not build or modify a
submission ZIP.

- baseline SHA-256:
  `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`;
- completed: **8/8**;
- accepted: **0**;
- all eight payloads are byte-identical to the previously audited 8004.50
  baseline;
- every compact Sakana true rule reproduces every known pair: task032 266/266,
  task041 266/266, task215 265/265, task211 266/266, task120 266/266,
  task235 69/69, task258 266/266, and task292 28/28;
- protected ZIPs, score authorities, CSVs, and shared artifacts were not
  modified.

## Per-task decision

| task | actual cost | incumbent observation | strongest lower/tie lead | decision |
|---:|---:|---|---|---|
| 032 | 70 | output-only 15-input Einsum, truthful runtime shapes | cost-46 six-row coefficient probe | **REJECT**: ORT broadcast failure on all 266 known cases in both modes |
| 041 | 70 | output-only 13-input Einsum, truthful runtime shapes | archive floors 74/78; clean floor 925 | **REJECT**: no actual strict decrease |
| 215 | 70 | prohibited 18-input giant Einsum, truthful runtime shapes | same-cost 70 rebuilds only | **REJECT**: no strictly-lower clean lineage |
| 211 | 66 | prohibited 22-input giant Einsum, truthful runtime shapes | cost-64 25-input giant Einsum | **REJECT**: candidate is 9/266 known in both modes and structurally forbidden |
| 120 | 64 | 15 runtime-shape contradictions; default ORT session fails | static-41 AveragePool proposal | **REJECT**: actual proposal cost is 2738 and it has seven shape contradictions |
| 235 | 64 | compact dynamic-Conv graph, truthful shapes, known dual 69/69 | clean spec rebuild 115; reduced 107/91 variants | **REJECT**: no graph below 64; reduced clean variants are wrong |
| 258 | 64 | shape-cloaked loss graph; default ORT session fails | clean grouped-Conv rule engine 160 | **REJECT**: no below-64 graph; clean floor is dominated |
| 292 | 64 | compact PRelu/CastLike/Einsum, truthful shapes, known dual 28/28 | cost-50/50/54 rank-one/sign-core graphs | **REJECT**: every strict lower graph is 0/28 known in both modes |

The inherited giant/shape-cloaked members remain fixed LB-white authorities;
they are not templates that authorize a new unsafe graph. Any replacement must
itself satisfy the current truthful-shape, dual-ORT, no-giant/no-cloak policy.

## Search evidence

1. The all-400 archive inventory covers **1,196 ZIPs, 448,568 ZIP members,
   233,751 loose observations, and 13,591 unique non-baseline graphs**. It
   retains numeric lower leads only for task032, task120, and task211; all three
   fail mandatory gates.
2. The focused 1,134-unique-model harvest finds no lower member for task041,
   task215, task235, task258, or task292. Apparent floors for those tasks are
   ties or above the current 64/70 costs.
3. Complete exact Wave2 finds no opportunity on any of the eight targets. The
   initializer-alias pass independently reports zero aliases and zero built
   candidates for all eight.
4. The task032 cost-46 candidate uses `Q[6,2]` against a static 30-row output;
   ORT rejects every known input in both modes. The task211 cost-64 candidate
   uses 25 Einsum operands and is only 9/266 correct in each mode.
5. Task120's archive entry is cheaper only under false static declarations:
   its measured runtime cost is **2738**, not 41. The current 64-cost member is
   itself shape-cloaked and default-ORT-invalid, so it cannot justify a new
   false-shape replacement.
6. For task292, the truthful sub-64 graphs at costs 50, 50, and 54 each fail
   all 28 known cases under both ORT modes. The decoded rule requires both a
   width-independent copy term and a modulo-3 color-change term, which the
   rank-one/sign-only families cannot express.
7. Existing spec-derived clean rebuild evidence gives dominated floors of 925
   (task041), 115 (task235), 160 (task258), and 80 for the older dense task292
   family. The compact cost-64 task292 incumbent has no exact initializer or
   node shave in the complete scans.

## Gate disposition

No proposal clears all prerequisite gates: strictly lower **actual** cost,
safe structure, truthful runtime shapes, and known100 in both ORT modes.
Therefore independent two-seed fresh-dual runs were not started. A fresh score
cannot make an actual-more-expensive, runtime-invalid, known-false, giant, or
shape-cloaked proposal adoptable. The 90% normal-task threshold and stricter
private-zero policy do not change this pre-gate rejection.

Authoritative machine-readable evidence:

- `baseline_audit.json` — hashes, actual costs, full checker, strict/data-prop,
  runtime shape traces, domains, fan-in inventory, and Conv-bias audit;
- `known_baseline_dual.json` — complete known runs in default and
  `ORT_DISABLE_ALL` modes;
- `true_rule_audit.json` — decoded rules and complete known-pair reproduction;
- `history_audit.json` — all-400 archive, focused harvest, exact passes, and
  scratch-report provenance;
- `lower_leads_dual.json` — reproducible actual costs and dual-ORT failures for
  every strict numeric lower lead;
- `result.json` — all eight final decisions;
- `winner_manifest.json` — empty authoritative promotion list.
