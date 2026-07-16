# C42 task379 result

## Winner

- Baseline: `baseline/task379.onnx`
  - SHA-256: `70a1ddaf15510384e8b4d88d1420e9639de9e36b1775468386216a2f723d0750`
  - memory 1570 + params 381 = cost 1951
- Candidate: `candidates/task379_qv_middle_rank2.onnx`
  - SHA-256: `4ddae903db9b2d4aceef3c501691b5cd2b862bc209f58f2e88969332c06dd455`
  - memory 1570 + params 379 = cost 1949
- Strict reduction: 2
- Projected score gain: `log(1951 / 1949) = 0.001025641115550439`

The repeated middle row of QV is represented by a 2x2x2 core.  Two distinct
3x2 one-hot maps reconstruct QV's `[q0,q1,q0]` mode and NV's `[q0,q1,q1]`
mode, while a two-element selector replaces the old three-element Rflip.
The old QV/NV/Rflip group costs 24 parameters and the exact factor costs 22.

## Gates

- Full ONNX checker: pass
- Strict shape inference with data propagation: pass
- Official-like known scoring: 266/266, errors 0
- Known differential, ORT disabled/default: raw 266/266 in both modes
- Fresh generator seed 800263379042, 5000 cases:
  - ORT disabled: raw 5000/5000, decoded 5000/5000, candidate errors 0
  - ORT default: raw 5000/5000, decoded 5000/5000, candidate errors 0
  - incumbent and candidate both score 4999/5000; the candidate adds no error
- Independent team validator random500: `ACCEPT_STRICT`
  - requested 500, executable 499, raw 499/499, threshold 499/499
  - skipped both failed 1, skipped one failed 0, mismatches 0
- Runtime shape trace, ORT disabled/default: 69 outputs, mismatches 0 in both
- Canonical `[1,10,30,30]` input/output, standard domain only
- No sparse initializer, nested graph, function, banned op, or Conv-bias issue

Evidence:

- `task379_qv_middle_rank2_final_audit.json`
- `task379_qv_middle_rank2_fresh5000.json`
- `task379_qv_middle_rank2_external500.json`
- `task379_qv_middle_rank2_external500_summary.json`

## Rejected probes

- Removing E was invalid: E's second row is the spatial coordinate basis, not
  an all-one redundant operand.
- Removing NV as an identity was invalid: NV has rank 2 and duplicates its
  second row; it is not the identity.
- Folding NV into M2 is algebraically exact in real arithmetic but changes the
  FP16 terminal Einsum contraction and fails raw/output checks.
- Threshold orientation classification fails because horizontal two-line
  variance spans values on both sides of 14/24; only vertical cases equal those
  exact values.
- The affine orientation basis activates extra FP16 terminal paths and fails
  raw/output checks because zero-times-overflow behavior is not preserved.

No shared submission ZIP, root CSV, score file, or artifact was modified.
