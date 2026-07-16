# Wave6 mid5 true-rule rebuild report

Assigned tasks: `task367`, `task209`, `task187`, `task023`.

Immutable campaign baselines supplied by the coordinator:

| task | baseline cost |
|---:|---:|
| 367 | 2197 |
| 209 | 2150 |
| 187 | 1814 |
| 023 | 1622 |

## Result

No candidate is eligible for integration. The healthy rebuilds all failed the
mandatory actual-cost-decrease gate. No ZIP or protected root file was changed.

| task | build | official measured cost | known | fresh | decision |
|---:|---|---:|---:|---:|---|
| 367 | truthful row-bit rectangle scan | 3915 | 266/266 | 2000/2000 | reject: cost 3915 >= 2197 |
| 209 | shape-cloak-free equivalent | 2372 | 266/266 | 1931/2000 (96.55%) in each ORT mode | reject: cost 2372 >= 2150 |
| 187 | shape-cloak-free equivalent | 56264 | 266/266 | not run after cost gate | reject: cost 56264 >= 1814 |
| 023 | no candidate promoted | — | — | — | reject: private-zero safety cannot be proved |

Official cost was measured with `scripts/lib/scoring.py::score_and_verify`, not
with a declared-shape-only estimate. The task209 and task367 measurements were
`memory=2117, params=255` and `memory=3540, params=375`, respectively.

## Safety evidence

- All three built controls pass full ONNX checker and strict shape inference
  with `data_prop=True`.
- The rebuilds contain no lookup table, private-zero shortcut, shape cloak,
  nested graph, banned op, or giant Einsum.
- Conv-family bias UB count is zero. Task209's QLinearConv nodes have no bias
  input.
- Task209 was run on the same 2000 generated cases with both
  `ORT_DISABLE_ALL` and default ORT: 69 semantic mismatches in each mode,
  0 runtime errors, and 0 margin violations. This is
  above the user's 90% policy threshold, but cost still prevents adoption.
- Task367's healthy rebuild is generator-exact on the tested 2000 fresh cases,
  but is 1718 cost units above the incumbent.
- The current task023 member is independently confirmed unsound at 1712/2000
  (85.6%). It was not reused. The compact Sakana decomposition rule is
  266/266 known but only 1996/2000 and 1991/2000 on two independent fresh
  streams. Because task023 has private-zero lineage, the required fresh 100%
  proof is absent, so neither the historical model nor this rule is eligible.

## Attempts and stop reason

Eight bounded attempts were exhausted: task023 local-square classification,
task023 recursive decomposition, task187 bounded-cross rule, task187 truthful
decloak, task209 truthful decloak, task209 packed-output structural reduction,
task367 bounded-cross rule, and task367 truthful row-mask scan. The compact
rules failed semantic gates; the exact/near-exact controls hit structural cost
floors above their incumbents. Further work in this lane would require a new
algorithm, not another safe trim.

Artifacts are diagnostic only:

- `candidate_task367_truthful_rowmask.onnx`
- `candidate_task209_decloaked.onnx`
- `candidate_task187_decloaked.onnx`
- `candidate_build_report.json`
- `baseline_runtime_trace.json`
- `reference_audit.json` (partial task023 audit before the cost-gate stop)
