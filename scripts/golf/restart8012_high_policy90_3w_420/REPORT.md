# cost 1001..2500 / 3-worker POLICY90 scan

Authority: `submission_base_8012.15.zip` (`1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231`)

Scope: 25 non-score25 tasks; excluded: 17; scanned: 8.

Workers: PIDs [87500, 87501, 87502]; candidate loads [69, 72, 78].

| task | authority | candidate | gain | class | known/fresh minimum |
|---:|---:|---:|---:|---|---:|
| 002 | 1286 | — | — | NO_ADMISSION | — |
| 005 | 2325 | — | — | NO_ADMISSION | — |
| 023 | 1321 | — | — | NO_ADMISSION | — |
| 054 | 2110 | — | — | NO_ADMISSION | — |
| 089 | 1340 | — | — | NO_ADMISSION | — |
| 125 | 1043 | — | — | NO_ADMISSION | — |
| 340 | 1173 | — | — | NO_ADMISSION | — |
| 367 | 2005 | — | — | NO_ADMISSION | — |

Conditional total gain: **+0.000000**
Conditional projected LB: **8012.150000**

The sole strict-lower structural lead was task023 at cost 1319 (authority
1321). It was known-exact but fresh accuracy was 85.95% / 84.00%, below
POLICY90, and was rejected.

POLICY90 candidates may be non-exact and are not an LB guarantee. All admitted
models pass checker, strict/static shapes, runtime-shape tracing, Conv-bias UB,
banned-op, nonfinite, output-shape, small-positive, and four-config gates.

The root submission, all_scores.csv, and others/ were not modified.
