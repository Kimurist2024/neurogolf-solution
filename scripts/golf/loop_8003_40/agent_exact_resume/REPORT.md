# Exact resume report (8003.40 baseline)

## Outcome

- Scanned: **400/400 tasks**
- Accepted: **0**
- ZIP merge: **not performed**
- Protected root/baseline files: **unchanged**
- Final verdict: **NO_SAFE_EXACT_CANDIDATE**

## Candidate decisions

| Task | SHA-256 | Cost | Evidence | Decision |
|---:|---|---:|---|---|
| 048 | `f9f597b2a8f575db6bdbd18b1183ec2861a2a1936389e74b383829bec22ccf4f` | 379 -> 378 | known 270/270; fresh 1818/2000 (90.9%), errors 0 | **REJECT**: below 95%; high-risk floating contraction change |
| 233 | `94cb98f706b7117336640949ca34ba92dc49c58a3bebd5708d90799ed160b123` | 7432 -> 7431 | known 266/266, errors 0; exact initializer alias | **REJECT**: highest-risk task and only +0.000134562 |
| 333 | `0628a573302f0a816d010482ed8b883caac7c307a27f47c9b53df85e2042a6bc` | 423 -> 421 | known 265/265; raw differential 2000/2000, errors 0 | **REJECT**: giant Einsum 36 -> 35 is platform-order sensitive |

Every candidate passes full checker, strict shape inference/data propagation, static-shape checks, banned-op checks, and the Conv/ConvTranspose/QLinearConv bias-length gate. Passing those structural gates does not override the risk/fresh policies above.

## Whole-baseline scan

- Byte-identical initializer dedup: 1 opportunity (task233)
- Exact disjoint outer fusion: 2 variants (task048)
- Exact sign/gauge absorption: 1 opportunity (task333)
- Overdeclared value_info proven by clean strict re-inference: 0 tasks
- Metadata scans completed: 400; strict re-inference unavailable for 14 baseline models
- task070: explicitly excluded as known private-zero lineage

The separate archive-resume lane is responsible for the externally sourced task109 annotation-only candidate; this lane did not merge or duplicate it.
