# Low44 target expansion — 8-task audit

## Outcome

All eight requested files were independently audited against immutable
`submission_base_8005.16.zip` (SHA-256
`73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`).
**No safe strictly-cheaper candidate exists in the complete retained history or
the applicable exact-rewrite scans.** The lane is complete at **8/8**, accepts
zero candidates, contributes `+0.0`, and does not build a ZIP.

The source true rules reproduce every known pair: task303 265/265, task098
266/266, task395 268/268, task167 268/268, task289 268/268, task038 266/266,
task262 11/11, and task269 266/266. All eight authoritative payloads are
byte-identical to the 8004.50 baseline. Protected ZIPs, score authorities, and
CSVs were not modified.

## Per-task decision

| task | actual cost | strongest lead | decision |
|---:|---:|---|---|
| 303 | 39 | one output-only 8-input Einsum; complete history has no sub-39 graph | **NO WINNER**: exact scans emit nothing |
| 098 | 37 | one output-only ConvTranspose with 27 kernel + 10 bias parameters | **NO WINNER**: no archive or exact sub-37 graph |
| 395 | 35 | current graph has two runtime-shape contradictions | **NO WINNER**: no sub-35 graph; inherited cloak cannot authorize a new candidate |
| 167 | 34 | truthful compact graph, 20 params + 14 actual memory | **NO WINNER**: no archive or exact sub-34 graph |
| 289 | 32 | remove leading Identity | **REJECT**: strict/full shape checking fails at CenterCropPad; current graph also fails default ORT and has four shape contradictions |
| 038 | 31 | no strict lower observation | **NO WINNER**: current graph has four shape contradictions and errors on every default-ORT known run |
| 262 | 31 | remove leading Identity | **REJECT**: strict/full checking exposes HannWindow shape 30 versus declared 1; current graph has two shape contradictions |
| 269 | 31 | remove leading Identity | **REJECT**: strict/full shape checking fails at CenterCropPad; current graph also fails default ORT and has four shape contradictions |

## Search evidence

1. The all-400 archive covers 1,196 ZIPs, 448,568 ZIP members, 233,751 loose
   observations, and 13,591 unique non-baseline graphs. It retains **zero**
   strict-lower models for every target in this lane.
2. The focused harvest contains 23 distinct target alternatives. Every one is
   tied or more expensive than the latest actual cost. The closest are exact
   ties for task098/task167/task289/task303 and cost32 versus the task038
   incumbent cost31.
3. Complete exact Wave2 finds only the three Identity removals above. Each is
   already fail-closed as a structural rejection: no candidate model survives
   full checker plus strict shape/data propagation. The independent all-400
   exact Einsum pass has zero target hits.
4. The truthful incumbents task303/task098/task167 are already at output-only or
   tiny-graph floors. Their exact scans find no unused/duplicate initializer,
   duplicate producer, Identity, or removable annotation.
5. The low nominal prices of task395/task289/task038/task262/task269 depend on
   false intermediate or output shapes. Reusing those declarations would fail
   the requested truthful-runtime-shape gate, even if known examples pass with
   `ORT_DISABLE_ALL`.

No proposal reaches the fresh gate because none first clears strict actual
cost, structural validity, truthful shapes, and dual-ORT known100. Running
fresh samples cannot make a non-cheaper or structurally invalid graph
admissible.

Machine-readable evidence is in `baseline_audit.json`, `true_rule_audit.json`,
`history_audit.json`, `result.json`, and `winner_manifest.json`.
