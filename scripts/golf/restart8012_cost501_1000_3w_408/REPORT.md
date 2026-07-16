# 8012.15 cost501..1000 three-worker wave 408

## Outcome

The lane found **1 POLICY90-admissible strict reduction(s)**.
The combined conditional gain is **+0.088293**, taking the immutable
8012.15 baseline to **8012.238293** if every listed candidate
is promoted.  This lane did not edit the root submission, score ledger, or
`others/`.

| task | authority | candidate | min known | min fresh | gain | candidate |
|---:|---:|---:|---:|---:|---:|---|
| 012 | 710 | 650 | 95.0943% | 94.5500% | +0.088293 | `scripts/golf/restart8012_cost501_1000_3w_408/candidates/task012_POLICY90_cost650_9aea31a6c01f.onnx` |

## Scope and exclusions

The immutable authority is `submission_base_8012.15.zip` with SHA-256
`1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231`.  All 31 non-score25 members whose current
cost is 501..1000 were enumerated.  The private-zero/unsound and known-LB-black
band members `[35, 66, 277, 198, 361, 251, 157, 182, 319]` were excluded before any
search.  The remaining 22 tasks were split
round-robin into three disjoint partitions of sizes
`[8, 7, 7]`: `[[34, 378, 237, 238, 46, 131, 382, 370], [363, 368, 19, 165, 107, 12, 280], [284, 69, 328, 117, 364, 201, 330]]`.

Each worker reprofiled every assigned current authority member, then ran all
three requested families: loose/ZIP archive strict-lower rebasing, current
graph exact simplification, and current cost<=10 pattern transfer.  Detailed
counters, rejected screens, authority profiles, and complete finalist runtime
rows are in `worker_0.json`, `worker_1.json`, and `worker_2.json`.

## Admission gates

Admission requires at least 90% whole-case accuracy independently in each of
four ORT configurations on complete known data and on 2,000 fresh examples
from each of two independent seeds.  Runtime errors, nonfinite outputs,
runtime/declaration shape mismatches, `(0,0.25)` positive outputs,
configuration sign disagreement, Conv-bias UB, nonstatic/cloaked shapes,
banned/nested graphs, sparse/functions, and nonstandard domains must all be
zero.  Candidate cost is independently profiled and must be strictly below
the current member.

This is a POLICY90 result, not an exact-correctness claim.  Root promotion and
leaderboard credit remain deliberately unclaimed.
