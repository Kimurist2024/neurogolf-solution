# others/71405 mid-cost lane 98 — LB supersession final

## Outcome

- New immutable authority: `submission_base_8008.14.zip`
- SHA-256: `50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6`
- MD5: `db4da5cc59186b26572a380725bc2fdf`
- Verified LB: **8008.14**
- Inspected: 11 files / 10 tasks
- Exact LB-white fixed: **10 files / 10 tasks**
- Pending probes: **0**

The exact candidate SHA for tasks 089/096/107/117/125-v2/138/156/157/165/209 is present in the 400-member LB-verified ZIP. Exact LB evidence supersedes the earlier local runtime, schema, and shape-cloak rejections. Those local diagnostics remain recorded because they explain why local policy alone was an unreliable LB oracle.

## Per-file disposition

| task | candidate SHA | old→candidate cost | local result | final result |
|---:|:---|:---:|:---|:---|
| 089 | `89183f12515c` | 1349→1340 | REJECT_RUNTIME_CONFIG | LB_WHITE_FIXED |
| 096 | `97f05f8495c7` | 1128→1123 | REJECT_RUNTIME_CONFIG | LB_WHITE_FIXED |
| 107 | `39e937f3065e` | 708→664 | REJECT_STRUCTURE_SCHEMA_UB | LB_WHITE_FIXED |
| 117 | `042e3ee0976a` | 606→605 | REJECT_STRUCTURE_SCHEMA_UB | LB_WHITE_FIXED |
| 125 | `d9af550bf535` | 1050→1048 | REJECT_RUNTIME_CONFIG | REJECT |
| 125 | `c30ac7a079a4` | 1050→1045 | REJECT_RUNTIME_CONFIG | LB_WHITE_FIXED |
| 138 | `55e71aec7157` | 2729→2705 | REJECT_SHAPE_CLOAK | LB_WHITE_FIXED |
| 156 | `e8b10010b50a` | 556→499 | REJECT_SHAPE_CLOAK | LB_WHITE_FIXED |
| 157 | `a1254f261940` | 853→849 | FRESH_PENDING | LB_WHITE_FIXED |
| 165 | `d6d40c11204c` | 592→587 | REJECT_RUNTIME_CONFIG | LB_WHITE_FIXED |
| 209 | `80c19164133e` | 2218→2087 | REJECT_SHAPE_CLOAK | LB_WHITE_FIXED |

The only nonmatching file is task125 `d9af550b...` (v1, cost1048). The verified ZIP contains task125 v2 `c30ac7a...` (cost1045), so v1 retains its local default-ORT rejection and is not separately probed.

## Local evidence retained

- Runtime-config rejects: tasks 089, 096, 125-v1, 125-v2, 165.
- Structural rejects: task107 negative Conv padding; task117 strict AffineGrid shape inference.
- Runtime-shape mismatches: task138 (36), task156 (1), task209 (16).
- task157 alone cleared the full local strict/UB, known265×4, and truthful-shape gates. Its expensive fresh2×500 run was stopped and not resumed because exact LB-white membership is stronger evidence.
- Private-high-risk tasks 096/138/157/209 are now fixed only because their exact SHAs are LB-white; this is not a task-level exemption for future versions.

## Score accounting

The ten member-level reductions sum to `0.272779156587`. The whole champion moved 8006.61→8008.14 (+1.53) because the verified ZIP contains 37 white changes in total, beyond this lane's ten.

## Artifacts

- `audit/pre_fresh_screen.json`: original competition profiles and local gates.
- `audit/authority_profiles_2x.json`: repeated 8006.61 authority profiles.
- `audit/lb_8008_14_supersession.json`: exact member comparison.
- `result.json`, `winner_manifest.json`, `probe_manifest.json`: final disposition.
