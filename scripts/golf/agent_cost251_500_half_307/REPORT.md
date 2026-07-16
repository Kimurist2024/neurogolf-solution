# Cost 251–500 half-cost / strict-improvement lane

## Outcome

One candidate survives the user's explicit `private-zero >=95%` exception:

| task | authority | candidate | gain | known, four configs | fresh, two seeds | class |
|---:|---:|---:|---:|---:|---:|---|
| 134 | 422 | **320** | **+0.276684318** | 266/266 each | 96.85%, 96.30% | `POLICY95_PRIVATE_ZERO_RISK_LOOKUP` |

It is staged at
`approved_policy95/task134_cost320.onnx`, SHA-256
`a610dcc58d2715ea4c39e000bfc83bb39ee69b69b95ee4a7ead252f3b126880b`.
It is not guaranteed safe: the graph uses ten `TfIdfVectorizer` lookup nodes
and task134 is a known private-zero lineage.  It is listed only because the
user expressly authorized these lineages when fresh accuracy is at least 95%.

There is no guaranteed-safe winner, and no admissible candidate reaches half
of its authority cost.

## Authority and rebase

The exhaustive scan began from `submission_base_8011.05.zip` (SHA-256
`ad96519a...`).  During the run the external authority advanced to
`submission_base_8012.15.zip`, SHA-256
`1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231`.
The task036, task134, and task268 authority members examined as finalists are
byte-identical across the two ZIPs.  The surviving task134 candidate was fully
rescored and rerun against 8012.15.

## Search coverage

- Scope at inventory time: all 61 authority tasks with cost 251–500.
- Cost<=10 transfer library: 148 generic/current-low-cost variants x 61 tasks
  = 9,028 quick evaluations; zero finalists.
- Loose history: 9,154 ONNX paths, 1,110 unique task/SHA pairs, 272 theoretical
  strict-lower/half candidates.
- ZIP history: 377 ZIPs, 22,580 target members, 900 unique task/SHA pairs, 219
  theoretical half candidates; zero known-exact actual half winners.
- Broad strict inventory: 975 unique artifacts; 621 passed static structural
  gates. Cost-first profiling found 136 provisional actual-cost reductions.
- One-config complete-known screening retained 36 artifact variants across
  seven tasks. Full history evidence and prior audits reduced these to the
  task134/036/268 deep-audit set; only task134 cost320 passed all requested
  policy gates.

The cost-first ordering is intentional.  The original known-first exhaustive
scanner was stopped after a pathological candidate spent more than a minute in
one ORT call.  Every structurally eligible artifact was still covered by an
isolated 12-second cost-profile subprocess, and complete-known candidates used
isolated 25-second subprocesses before deep audit.

## task134 evidence

- Full checker and strict data-propagating shape inference pass.
- Canonical static input/output shape `[1,10,30,30]`.
- Runtime trace mismatch count 0; shape cloak false.
- Standard domains, no banned op, no local function/nested graph/sparse init.
- Finite initializers; Conv-family bias UB 0.
- Margin stable; minimum positive 1.0; no value in `(0,0.25)`.
- Disabled/default ORT x threads 1/4: 266/266 known in every configuration.
- Seeds `307001341` and `307001342`, 2,000 cases each: 1,937/2,000 and
  1,926/2,000 in every configuration; zero errors, nonfinite values, output
  shape failures, or small-positive values.

The cheaper task134 cost319 variant was also checked: known 264/266 and fresh
94.70%/94.75%, so it fails the 95% gate.

## Decisive rejections

- task168 414→166: fresh 607/2,000 = 30.35%.
- task048 379→142: fresh 611/1,000 = 61.10%.
- task036 provisional 307→287 was a one-case dynamic-profile underestimate;
  the artifact is byte-identical to the authority, full delta is 0, and some
  fresh outputs have noncanonical shapes.
- task268 420→327: disabled fresh 41.35%/45.10%; default ORT cannot construct
  the session.
- task102 493→491 and task200 346→344 use Conv-family bias-shape undefined
  behavior and remain rejected.
- task185 lower variants are known-perfect but existing independent fresh
  evidence is about 0.2%, far below policy.
- task199/task333 latent deletions score 0/266 and at most 1/265 known cases,
  respectively.

## Files

- `candidate_manifest.json`: authoritative lane handoff.
- `task134_cost320_rebase8012_policy95_audit.json`: four-config/two-seed audit.
- `task134_runtime_structure.json`: runtime shape and structural audit.
- `lowcost_transfer_evidence.json`: transfer-library results.
- `history_half_evidence.json`, `zip_half_evidence.json`: half-cost history.
- `strict_inventory.json`, `cost_first_rank.json`, `known_first_rank.json`:
  complete strict-improvement funnel.

No root ZIP, score ledger, `others/`, or documentation outside this lane was
modified by this worker, and no merge was performed.
