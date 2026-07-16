# Low-cost target expansion — 8-task audit

## Outcome

The requested eight members were independently re-audited against
`submission_base_8005.16.zip`. **No safe strictly-cheaper candidate exists in
the searched exact/history/SOUND families.** This lane contributes **+0.0** and
does not build or integrate a ZIP.

- latest archive SHA-256:
  `73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`;
- completed: **8/8**;
- accepted: **0**;
- all eight member bytes are unchanged from `submission_base_8004.50.zip`, so
  every conclusion is rebase-compatible with 8005.16;
- protected ZIPs, score files, CSVs, and shared handcrafted models were not
  modified.

## Per-task decision

| task | current actual cost | strongest lower/tie lead | decision |
|---:|---:|---|---|
| 221 | 144 | cost-142 shape-cloak edit | **REJECT**: ORT_DISABLE_ALL buffer-shape failure on the complete known set; correct controls start at 151 |
| 136 | 135 | cost-135 history tie | **REJECT**: no decrease; 58-input giant Einsum and false runtime shapes; conventional controls start at 1194 |
| 278 | 135 | cost-135 Min/Max tie | **REJECT**: no decrease and complete-known runtime failure under ORT_DISABLE_ALL; conventional history is larger |
| 230 | 108 | cost-108 family / cost-900 direct Conv | **REJECT**: no decrease; the low-cost family has dynamic Conv bias whose UB0 cannot be proved |
| 327 | 106 | cost-106 archive tie; proposed cost-46 one-node net | **REJECT**: archive only ties; the cost-46 architecture is mathematically infeasible over the complete placement domain |
| 391 | 104 | cost 85/87/88 lookup nets | **REJECT**: all are `TfIdfVectorizer` lookup/private-zero lineage; smallest table-free true-rule engine costs 139 |
| 097 | 100 | cost-100 local controls | **REJECT**: all valid history ties 100 or costs 910; no strict decrease |
| 027 | 96 | cost-96 alternate giant-Einsum net | **REJECT**: tie only and still a 55-input giant Einsum; conventional control costs at least 962 |

The current low-cost members are LB-white authorities, not templates that
authorize new unsafe structures. Runtime tracing found declared/actual shape
mismatches in tasks 221, 136, 278, 230, 327, and 097. Task027 contains a
55-input giant Einsum. Task391 is runtime-shape truthful, but its lookup payload
is grandfathered by its verified LB lineage; it does not authorize a new lookup
replacement.

## Search evidence

1. The exact Wave2 audit scanned all 400 members of the byte-identical 8004.50
   base and accepted zero. None of these eight tasks produced an exact rewrite;
   task230 appears only as a conservative Conv-bias structural failure.
2. The broad archive inventory covered 1,196 ZIPs, 448,568 ZIP members,
   233,751 loose files, and 13,591 unique non-baseline graphs. For these eight
   tasks, only task391 retained any below-baseline history, and all five retained
   graphs are forbidden lookup models.
3. The focused harvest confirms task027 and task097 only have same-cost lows,
   task230's same-cost graph has a dynamic bias, and task327's best archived
   rebuild only ties cost106.
4. For task327, the archived one-node cost-46 `ConvTranspose` proposal was
   rerun against all 51 legal generator placements. It fails before LP solving:
   the same local binary patch and the same bias kind require labels 0 and 1.
   Therefore no affine one-node model in that proposed family can exist, and no
   candidate file was emitted.
5. For task391, the true rule was previously verified on 267/267 known and
   5000/5000 fresh cases. This does not rescue the cheaper lookup graphs: the
   cost-102 lookup replacement is documented private-zero, while the truthful
   table-free TopK implementation costs 139 because its intermediate memory
   alone is 104 bytes before 35 parameters.

## Gate disposition

No model cleared both prerequisite gates: strictly lower actual cost and safe
structure (truthful runtime shapes, standard domains, no lookup/cloak/giant,
and Conv-bias UB0). Consequently known-dual and the two independent fresh-dual
runs were not started. Running fresh tests cannot make a same-cost or
structurally forbidden model adoptable.

Authoritative machine-readable evidence:

- `baseline_audit.json` — latest member hashes, actual costs, strict checks,
  runtime-shape traces, and exact-Wave2 cross-check;
- `history_audit.json` — archive/harvest evidence, task327 infeasibility, and
  task391 SOUND/private-zero controls;
- `evidence/task327_one_node_infeasible.json` — explicit conflicting local
  constraints;
- `result.json` — all eight final decisions;
- `winner_manifest.json` — empty authoritative promotion list.
