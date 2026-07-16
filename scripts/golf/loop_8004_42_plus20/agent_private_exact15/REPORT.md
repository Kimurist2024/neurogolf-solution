# Private-zero exact micro-rewrite audit

## Result

- Baseline archive: `submission_base_8004.50.zip`
- Baseline archive SHA-256: `63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`
- Audited tasks: `077, 102, 169, 187, 191, 216, 285, 366`
- Accepted candidates: **0**
- Verified projected gain: **+0.0**
- Final verdict: **NO_GUARANTEED_PRIVATE_EXACT_CANDIDATE**

The exact task members extracted from the LB-white archive were SHA-checked
before every rewrite.  No root submission, handcrafted artifact, or incumbent
file was modified.  Rejected provisional models were deleted; the `candidates/`
directory is empty.

## Admission policy used

A private-zero-lineage rewrite was admissible only if all of the following
passed:

1. the source bytes were the exact current `8004.50` member;
2. the rewrite had a formal all-input ONNX-semantic equivalence argument;
3. full checker and strict shape inference with `data_prop=True` passed;
4. all declared static shapes were positive and matched runtime shapes;
5. standard domains, no nested/function/sparse/external payloads, banned-op
   absence, finite initializers, and Conv-family bias UB=0 passed;
6. measured cost was strictly lower;
7. every known case was correct and raw-bitwise identical to the baseline in
   both `ORT_DISABLE_ALL` and default ORT;
8. only after all pre-fresh gates passed: 5,000 fresh cases under each of two
   independent seeds, in both ORT modes, had to be 100% correct and raw-bitwise
   identical with zero runtime errors.

No candidate reached step 8, so fresh generation was correctly skipped.

## Candidate decisions

| Task | Exact source SHA-256 | Rewrite | Cost evidence | Runtime / structural evidence | Decision |
|---:|---|---|---|---|---|
| 077 | `db46560f4e633e057f960fe2db040f62b118051fc1bbbfc6e29871fcd0e84d56` | remove Identity 353 | incumbent measured cost 3364; candidate cannot be fully scored because strict inference fails | known `266/266` correct and raw-bitwise equal in both modes, but candidate retains 10 declared/runtime mismatches; exposing the literal shape makes strict `CenterCropPad` inference contradict its annotations | **REJECT** |
| 102 | `48d974a3aa19e409784595ec9ce076d5168974171abff37183165a918ea5e867` | remove Identity 0 and 3 | incumbent measured cost 493; candidate cost unavailable after mandatory structural failure | combined candidate and both single-removal probes fail strict `CenterCropPad` shape inference and cannot create either ORT session; incumbent itself has 55 shape mismatches | **REJECT** |
| 169 | `580aeebba96eacc482a11e3a6da6f4295758ab24e2376a8a248327f943cb33b4` | merge duplicate deterministic Selu 12 into 11 | static 248 -> 246, nominal `+0.0080972102` | incumbent has 107 shape mismatches; candidate hits allocator/buffer shape mismatch on all 266 known cases under disabled ORT and fails default-session construction | **REJECT** |
| 187 | `20f103296641dacfbb5ff424b60cf3006c9af42dba6ada7e985576609a2100b0` | remove Identity 7 | static 1582 -> 1574, nominal `+0.0050697194`; incumbent measured cost is 1814 | incumbent has 13 shape mismatches; candidate cannot create either ORT session because the now-visible literal `TopK` request exceeds the inferred axis | **REJECT** |
| 191 | `109928c1f7ec9fd2ca497bcc538bd0a7065cc5fda572855b0ac7a12299c3c115` | remove Identity 0 | incumbent measured cost 897; candidate cost unavailable after mandatory structural failure | strict `CenterCropPad` inference fails, both ORT sessions fail, and incumbent has 36 shape mismatches | **REJECT** |
| 216 | `bc62a647f05d5b736e0e2b4443ff2e3d63224e43f6f015993750f7c0e04b4abe` | remove Identity 11 | incumbent measured cost 1037; candidate cannot be fully scored because strict inference fails | known `266/266` correct and raw-bitwise equal in both modes, but candidate retains 50 shape mismatches; the literal shape exposes strict `Slice` / `CenterCropPad` contradictions | **REJECT** |
| 285 | `366212e29105fde0295030f3ec3bb014bd300f23aa8259ccd79da2eea720b9e2` | remove Identity 204 | incumbent measured cost 8623; candidate cost unavailable after mandatory structural failure | strict inference fails; disabled ORT candidate has runtime errors on all 265 known cases, default ORT session construction fails, and incumbent has 55 shape mismatches | **REJECT** |
| 366 | `072ca4f43fb6fe6e96cf90826d5f9b2dbdbe5016f7db3daa14f265620c1a010a` | remove Identity 81 | incumbent measured cost 7987; candidate cannot be fully scored because strict inference fails | known `255/255` correct and raw-bitwise equal in both modes, but candidate retains 98 shape mismatches; the literal shape exposes four strict `CenterCropPad` contradictions | **REJECT** |

The task077/216/366 rewrites are algebraic Identity eliminations and sampled
outputs are exact, but that is not an all-input execution guarantee for these
models: the Identity is also a data-propagation barrier inside an invalid
shape-annotation lineage.  Removing it changes ORT inference/liveness facts,
and closely related task169/187/285 probes demonstrate concrete allocator or
session failures.  Therefore these three cannot be promoted under the user's
"guaranteed private-zero" exception.

## Evidence

- Machine-readable aggregate: `result.json`
- Reproducible builder/auditor: `audit_exact.py`
- Per-task evidence: `task077_audit.json`, `task102_audit.json`,
  `task169_audit.json`, `task187_audit.json`, `task191_audit.json`,
  `task216_audit.json`, `task285_audit.json`, `task366_audit.json`
- Separate task102 single-removal evidence:
  `task102_identity0_audit.json`, `task102_identity3_audit.json`
