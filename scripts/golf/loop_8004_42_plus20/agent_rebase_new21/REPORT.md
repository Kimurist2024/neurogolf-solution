# 8005.16 rebase-new21 audit — 8 tasks

## Outcome

採用候補は **0件**、このレーンの加点は **+0.000000** です。`submission_base_8005.16.zip` と root の提出物・CSV・`best_score.json` は変更していません。

| task | incumbent cost | truthful runtime shapes | fresh | decision | main blocker |
|---:|---:|:---:|:---|:---:|:---|
| 013 | 638 | True | - | REJECT | 51-input terminal Einsum violates the no-giant-contraction gate; the truthful spec control costs 7884, above the cost-638 incumbent |
| 018 | 4753 | False | - | REJECT | strict data-propagating shape inference fails; default ORT session creation fails |
| 054 | 2258 | False | - | REJECT | default ORT session creation fails; 40 declared/runtime shape mismatches |
| 080 | 3050 | True | 5000/5000 ×2 seeds ×2 ORT | REJECT | no strictly cheaper candidate survived exact scanning; no Identity, duplicate producer, unused initializer, or further safe scalar-broadcast shave exists |
| 089 | 1349 | False | - | REJECT | default ORT session creation fails; 49 declared/runtime shape mismatches |
| 096 | 1128 | False | - | REJECT | default ORT session creation fails; 6 declared/runtime shape mismatches |
| 101 | 5655 | True | 994/1000, 987/1000 ×2 ORT | REJECT | private-zero lineage requires 100% fresh correctness; fresh correctness is 994/1000 and 987/1000 in both ORT modes |
| 131 | 691 | False | - | REJECT | 18 declared/runtime shape mismatches; the independent spec rebuild costs 1521, above 691 |

## Important independent re-audit: task089 root lead

- Candidate: `scripts/golf/loop_8004_42_plus20/root_exact_noop26/task089.onnx`
- SHA-256: `d8793c28527b54545d1cc504a7bfec11cda0c048cdf28bef1f12241889a55eb1`
- Apparent cost: `1349 -> 1180` (`+0.133849138748` if valid)
- Actual gate: **REJECT**. ORT_DISABLE_ALL gives candidate correct `0/267` with runtime errors `267/267`; default ORT session creation fails. Raw equality to the LB incumbent is `0/267`. Runtime-shape contradictions: `49`. UB findings: `0`.
- Fresh was deliberately not run after the mandatory known/runtime/truthful gates failed.

## Only structurally healthy incumbent: task080

task080 is checker/strict/data_prop/static/truthful/UB0 clean and remains spec-derived. It passed `5000/5000` on each of two independent fresh seeds in both ORT modes (20,000 executions total), with runtime errors 0 and dual raw equality 100%. However, exact scan found no cheaper graph. The closest prior ablations cost 3073 (already above 3050) and fail stored gold.

## Other decisive blockers

- task013 is shape-truthful and known-perfect but terminates in a **51-input Einsum**, violating the explicit no-giant-contraction gate. The non-giant spec control costs 7884 versus 638.
- task018/054/089/096/131 have declared/runtime shape contradictions (61/40/49/6/18 respectively); several also fail default ORT. Truthful controls are all more expensive.
- task101 is structurally clean but is private-zero lineage. It scored only 994/1000 and 987/1000 on independent fresh seeds in both ORT modes, below the required 100%; the sound decoded-rule control costs 7264 versus 5655.
- task096's only apparent exact Identity prune is not valid: it exposes the shape carrier and then fails checker/ORT initialization.

## Evidence

- `baseline_audit.json`: complete known dual-ORT, cost, strict/data_prop, runtime-shape and UB audit for all 8 incumbents.
- `fresh_audit.json`: task080 and task101 independent fresh dual-ORT runs.
- `task089_root_candidate_audit.json`: the cost-1180 root lead rejection.
- `result.json`: machine-readable final decisions and exact SHA values.

Final decision: **do not merge any model from this lane**.
