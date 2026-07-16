# High-gain lane report — latest LB 8004.50

## Outcome

No candidate passes every safety and cost gate. This lane contributes
**+0.000000** projected score, builds no ZIP, and changes no protected file.

The authority is `submission_base_8004.50.zip`, SHA-256
`63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`.
The earlier 8004.42 fixed-rebase ZIP is not used: the LB-verified 8004.50 ZIP
already contains the eight safe rebase changes, while task344 is excluded.

## Primary high-gain targets

| task | 8004.50 cost | best relevant lead | possible gain | decision |
|---:|---:|---:|---:|---|
| 145 | 5130 | sound rebuild 10175 | negative | no cheaper sound graph |
| 191 | 3444 | 3426 | +0.005240 | default-ORT/session and shape-cloak reject |
| 204 | 2240 | sound history 2544 | negative | no cheaper safe full model |
| 205 | 1042 | 937 | +0.106214 | 4904/5000, worse than baseline 4928/5000 |
| 285 | 8623 | sound rebuild 14699 | negative | no cheaper sound graph |

task205 is the only primary lead above `+0.05`. It is complete on all 266
known cases and has zero generator-domain runtime errors, but the independent
dual-ORT seed gives 4904/5000 versus 4928/5000 for the current member. The
candidate adds 24 fresh mistakes, so it remains comparison-only. Its
13-input float contraction is also higher-platform-risk than the current
member. No candidate was copied into this lane's `candidates/` directory.

## Archive coverage and terminal reasons

The SHA-deduplicated primary scan collected 645 models with no read errors:
93 task145, 40 task191, 47 task204, 229 task205, and 236 task285. The complete
per-SHA static classification is `static_inventory.json`. Very low "costs"
there are optimistic static floors, not actual scorer costs; many are partial
builder probes or graphs whose runtime tensors are much larger than declared.

The execution-relevant frontier families have already been independently
audited under both ORT modes in the retained 7999.13/8002.63 evidence:

- task145: archive and crop-fusion variants are known-wrong, runtime-invalid,
  or shape-cloaked; the honest 3000/3000 graph costs 10175.
- task191: the only actual cheaper models cost 3426/3430, both fail the
  default-runtime gate, and save much less than +0.05.
- task204: the current 2240 member is already below every prior full model;
  the old 2544 rewrite is no longer an improvement.
- task205: cost 459 is shape-invalid; 937 is the weaker 98.08% lead; costs
  951/965/997 and the four cost-977 rebuilds use 20-input floating Einsum and
  have retained fresh failures; 1010/1015 are private-zero lineage; 1036 is
  runtime-invalid.
- task285: tiny files under the task directory are component/operator probes,
  not full rule networks. All runnable historical full models are more
  expensive or heuristic/shape-cloaked; the honest rule engine costs 14699.

Every archived SHA remains recorded even when it is rejected at the static
floor, structural, partial-probe, lineage, runtime, correctness, or fresh gate.

## Additional high-cost sweep

Nine more current high-cost tasks were checked against the retained strict
audits. None yields an eligible `+0.05` lead:

| task | current | best lead | optimistic gain | terminal rejection |
|---:|---:|---:|---:|---|
| 366 | 7987 | 7646 | +0.043633 | 107 shape contradictions; 98.4864% fresh |
| 286 | 7481 | 7122 | +0.049178 | fixture correction lookup; 85.88% fresh |
| 233 | 7432 | sound 17007 | negative | no cheaper clean implementation |
| 101 | 5712 | 5672 | +0.007027 | input-signature fixture branch |
| 018 | 4754 | 4733 | +0.004427 | default ORT and shape-cloak failure |
| 133 | 4403 | sound 5570 | negative | no cheaper clean implementation |
| 118 | 3665 | none | — | generator relation is non-deterministic |
| 173 | 3525 | 3513 | +0.003410 | scalar-declared full-grid output cloak |
| 077 | 3364 | 3345 | +0.005664 | ten shape contradictions; 4807/5000 |

## Evidence map

- `RESULTS.json`: final dispositions and exact gains;
- `static_inventory.json`: every collected primary SHA and static reason;
- `collection.json`: collection counts and 8004.50 member hashes;
- `evidence/task205_vs_8004_50_random100.json`: latest-base known/cost and
  arbitrary-input differential;
- `evidence/task205_static_shape.json`: full checker, strict data propagation,
  truthful runtime shapes, and zero Conv-bias findings for the sole >0.05 lead;
- `run_targeted_scan.py`, `classify_static.py`: reproducible non-promoting
  scanners;
- prior strict evidence:
  `loop_7999_13/lane_a20`, `lane_a21`, `lane_a22`, `lane_a23`, `lane_a19`,
  `lane_c11`, `lane_c13`, `lane_c17`, `lane_c19`, `lane_c20`, `lane_c41`, and
  `loop_7999_13/lane_a41`.

The protected submission ZIPs, score ledgers, and root CSVs were not modified.
