# Exact-safe lane report (8004.42 fixed rebase)

## Outcome

- Immutable base: `submission_8004.42_fixed_rebase_meta.zip` (`dc64ef5a...e0149`).
- Fixed LB/rebased tasks preserved: all 27 were excluded from rewrites.
- Accepted candidates: **0**.
- Projected accepted gain: **+0.000000**.
- ZIP merge: **not performed**.
- Root score/submission files: **not changed**.

## Runtime-shape rejections

Six algebraically exact candidates passed ONNX checker and strict shape inference but
failed the runtime-error gate. Tasks 039, 089, 111 and 122 removed output-unreachable
nodes; tasks 165 and 169 eliminated duplicate producers. Each exposed ORT `Slice`
buffer-reuse shape errors, so all six are rejected regardless of potential savings.

## task233 decision

Candidate:
`scripts/golf/loop_8004_42/agent_exact_safe/models/task233.onnx`

- SHA-256: `94cb98f706b7117336640949ca34ba92dc49c58a3bebd5708d90799ed160b123`
- Rewrite: alias `audit_one_i16` to byte-identical `one_i8`; no floating operation or
  contraction order changes.
- Cost: 7432 -> 7431; projected gain `+0.000134562336`.
- Known evidence: 266/266, runtime errors 0.
- Fresh dual probe: target `ORT_DISABLE_ALL` 10/10 correct and raw bitwise 10/10;
  default ORT raw bitwise 10/10 but only 1/10 ground-truth correct. Runtime errors 0.
- Structural gates: checker PASS, strict inference/data propagation PASS, static
  shapes PASS, banned/nested ops 0, Conv-family bias UB 0.

**Verdict: REJECT.** The full 2000-3000 fresh adoption gate was not completed,
default-mode fresh accuracy was below the dual-mode safety bar, and task233 is a
highest-risk monitored task for only a dust-sized gain.

## Other screen

Representative latent-axis reductions for tasks 010, 028, 060, 163, 175, 199,
229, 232, 304 and 315 all failed the known-complete verifier and were rejected.

Final verdict: **NO_SAFE_EXACT_CANDIDATE**.
