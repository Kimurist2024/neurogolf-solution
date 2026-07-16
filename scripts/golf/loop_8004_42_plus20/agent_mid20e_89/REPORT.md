# mid20e_89 — fifth 20-task expansion, LB 8006.61 rebase

## Result

No locally admissible score improvement was found. The fixed-adoption manifest
is empty and projected accepted gain is `0.0`. No submission ZIP, score CSV,
ledger, root pointer, or `others/71403` content was modified.

The authority is `submission_base_8006.61.zip`, SHA-256
`9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`
(LB 8006.61).

## Authority rebase proof

All 20 target members are byte-identical between `submission_base_8005.17.zip`
and `submission_base_8006.61.zip`. `authority_rebase_proof.json` records both
member hashes and sizes for tasks 020, 030, 059, 068, 175, 183, 190, 193, 195,
224, 240, 281, 300, 302, 304, 358, 376, 383, 384, and 400.

The fixed champions 013/070/158/254/267/323/379 have no intersection with this
target set and were not touched. The current net-specific LB-probe-required set
018/048/112/134/168/198/233/251/277/286/365/366 also has no target
intersection. Those labels are treated as SHA/net-specific evidence, not as a
permanent ban on future true-rule implementations.

## Exhaustive history scan

Every loose ONNX and matching member of every repository ZIP was inventoried
and SHA-deduplicated against the new authority. The scan found 738 distinct
non-authority SHAs:

- static cost reject: 408
- structural reject: 185
- actual cost reject: 90
- policy reject: 53
- official-known reject: 2

The 53 policy rows were not left as blanket rejects. Nineteen use schema-invalid
negative Conv padding and remain hard rejects. The other 34 are task302
private-lineage SHAs and were reopened as net-specific candidates: 33 have a
static floor at or above authority cost 151, proving they cannot be
strict-lower. The remaining SHA `b4812bd2...` had a nominal static floor 150
and was fully profiled. Its real cost is 52,346, it has 76 declared/runtime
shape mismatches, and default ORT cannot create a session. It is therefore a
shape-cloak/runtime reject, not an LB-probe candidate. `probe_manifest.json`
contains zero candidates.

Ninety-two candidates reached isolated runtime cost profiling. Only task193
SHA `82ebe943...` at cost 100 and task384 SHA `d4f13184...` at cost 179 were
strict-lower after that stage. Both were run through four complete known passes
(two new sessions in each ORT mode), with zero runtime errors:

- task193: 154/266 correct in all four runs;
- task384: 265/266 correct in all four runs.

Neither passes the mandatory complete-known gate, so neither is a fixed
adoption or an LB-probe candidate under this lane's requirements. No candidate
reached fresh validation; the two-seed fresh threshold cannot rescue a known
failure.

## Authority caveat and exact reductions

The task059 authority member itself is 0/266 on four known runs in both ORT
modes. No strict-lower sound historical candidate exists. Consequently,
behavior-preserving trims of this unsound authority would not constitute a
safe improvement and were fail-closed.

Task302's historical private label was treated at SHA/net level, not as a
permanent task exclusion. Its 34 policy SHAs were reopened as described above;
none survives actual-cost and shape/runtime gates. Lookup/giant, shape-cloak,
non-static, schema-invalid negative Conv pads, and Conv-family bias UB remain
evidence-based rejects.

The exact authority members were additionally checked for unused or duplicate
initializers, dead/neutral operands, removable zero Conv-family biases,
bypassable Identity nodes, identity Cast/Reshape/Transpose, single-input
Concat, and zero Pad. No candidate transformation exists across these 20
members.

## Evidence

- `authority_rebase_proof.json`: all 20 byte-identity proofs and protected-set intersections
- `result.json`: final re-based decision
- `winner_manifest.json`: empty fixed-adoption manifest
- `rescreen.json`: all 738 SHA rows against 8006.61
- `inventory/raw.json`: raw inventory metadata
- `inventory/summary.json`: per-task terminal results
- `audit/candidate_known_quad.json`: four-run evidence for tasks193/384
- `probe_manifest.json`: LB-probe-required candidate manifest (empty)
- `audit/policy_reopen.json`: all 53 policy rows and reopened decisions
- `audit/task059_authority_known_quad.json`: authority unsoundness evidence
- `audit/known_rejections.json`: official-known decisions
- `mechanical_reductions.json`: exact/no-op audit
