# High-cost history pre-screen 49

## Outcome

Eight previously unreported 8005.16 members and every retained numeric lower
history lead were re-profiled with official-like runtime cost and complete-known
execution in both ORT modes. No candidate is admissible. Accepted: **0**;
projected gain: **+0.0**. No submission archive or protected score file changed.

| task | base actual | decisive lower-lead result |
|---:|---:|---|
| 037 | 374 | static322 is actual437668 and default-invalid; static334/372 fail all known cases at runtime |
| 297 | 371 | actual361 is known265/265 dual and otherwise clean, but uses out-of-schema negative Conv pads `[0,0,0,-24]`; standard Slice/Split repairs cost484/511 |
| 014 | 203 | static156/160/170 leads profile at 1122–1154; static192 profiles at201 but fails every known case under ORT_DISABLE_ALL and is shape-cloaked |
| 092 | 366 | all five lower-static leads profile at actual393–418 and are shape-cloaked/default-invalid |
| 398 | 347 | three actual332 leads are known0/268 and retain giant Einsums |
| 218 | 329 | four leads cannot be profiled or executed; the actual314 lead is known0/266 and shape-cloaked |
| 132 | 312 | actual282/287 leads are all known0/267 and giant-Einsum models |
| 388 | 91 | apparent static81/82 leads profile at 9089/9090; remaining candidates tie or exceed 91 and are shape-cloaked |

The task297 candidate is not rescued by its known correctness: ONNX Conv pads
are nonnegative by schema, and the previously built standard-domain repairs are
strictly more expensive. Fresh testing cannot make an out-of-schema candidate
admissible, so it remains rejected before that gate.

Machine-readable evidence is in `history_lead_audit.json`; the earlier complete
task297 schema proof is `scripts/golf/loop_7999_13/lane_a10/task297_audit.json`
and `failure_manifest.json`.
