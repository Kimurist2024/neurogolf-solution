# Low43 target expansion — 8-task audit

## Outcome

The requested eight members were independently audited against the immutable
`submission_base_8005.16.zip`. **No safe strictly-cheaper candidate survived.**
This lane contributes **+0.0**, emits no winner, and does not build or modify a
submission ZIP.

- baseline SHA-256:
  `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`;
- completed: **8/8**;
- accepted: **0**;
- all eight baseline members are byte-identical to the 8004.50 base;
- every baseline is 100% correct on the complete known corpus in both default
  ORT and `ORT_DISABLE_ALL`, with zero near-margin cells;
- the decoded Sakana rule reproduces every known pair for all eight tasks;
- protected ZIPs, score files, CSVs, and shared models were not modified.

## Per-task decision

| task | actual cost | strongest lower/tie lead | decision |
|---:|---:|---|---|
| 006 | 45 | history costs 30/38/40/40 | **REJECT**: complete-known dual results are 0/266, runtime failure, runtime failure, and 27/266; truthful spec rebuild floor 56 |
| 334 | 43 | focused-history floor 85 | **REJECT**: no strict decrease in archive, focused harvest, or exact passes; prior supported spec rebuild floor 49 |
| 244 | 41 | different-history floor 43 | **REJECT**: all history is more expensive; safe spec rebuild 70 and its 67 shortcut is generator-wrong |
| 249 | 41 | cost-41 tie, then 206 | **REJECT**: tie only; supported spec construction costs 57 |
| 347 | 41 | different-history floor 51 | **REJECT**: all history is more expensive; truthful rebuild costs 56 and lower grouped routing is impossible for source colors 3/4 to output color 6 |
| 386 | 41 | different-history floor 60 | **REJECT**: all history is more expensive; truthful rebuild costs 78 and lower direct/grouped families fail the output contract |
| 146 | 40 | two apparent static-cost-38 graphs | **REJECT**: both reprofile to actual cost 67, declare false runtime shapes, and fail default ORT session creation |
| 291 | 40 | cost-30 channel-free Einsum | **REJECT**: 0/265 known in both ORT modes; sign[10] plus edge[30] is the truthful zero-intermediate cost-40 floor |

## Search evidence

1. The all-400 archive covers **1,196 ZIPs, 448,568 ZIP members, 233,751
   loose observations, and 13,591 unique different graphs**. Its only numeric
   leads are the rejected task006 and task146 families above.
2. The independent focused harvest covers another **1,134 unique different
   graphs**. It provides no adoptable below-baseline member for these targets.
3. Exact Wave2 scanned all 400 baseline members and reports zero target hits.
   The independent low43 scan also finds no Identity, unused or duplicate
   initializer, duplicate producer, or other exact price-reducing local edit.
4. The all-400 exact Einsum scan reports no initializer-dedup, outer-fusion, or
   sign-absorption hit for any low43 target.
5. The decoded rules cover binary intersection (006), fixed glyph selection
   (334), magnified line-grid decoding (244), horizontal duplication (249),
   binary union (347), binary NOR (386), asymmetric-block selection (146), and
   hollow-object color selection (291). They reproduce **100% of all known
   pairs**, so rejection is not based on visible-example guessing.
6. Tasks006/347/386 in the immutable LB-white baseline have runtime annotation
   contradictions. Those exact bytes remain fixed; their lineage is not used
   to authorize a newly modified unsafe candidate.

## Gate disposition

No proposal clears all prerequisite price, known-dual, runtime, and truthful
structure gates. Therefore two-seed fresh validation is deliberately not run
for a promotion candidate: fresh accuracy cannot rescue a tie, a graph that is
more expensive after runtime accounting, a known-corpus failure, or a runtime
shape cloak.

Authoritative machine-readable evidence:

- `baseline_audit.json` — hashes, actual costs, checker/full shape inference,
  runtime shape traces, domains, op inventory, and Conv-bias audit;
- `known_baseline_dual.json` — complete-known baseline results in both ORT
  modes;
- `true_rule_audit.json` — decoded rules and all-known-pair reproduction;
- `history_audit.json` — archive, focused harvest, and exact Wave2 evidence;
- `lower_leads_dual.json` — actual reprofiling and complete-known dual results
  for every numeric lead;
- `exact_candidate_scan.json` — independent exact-rewrite scan;
- `result.json` — all eight final decisions;
- `winner_manifest.json` — empty authoritative promotion list.
