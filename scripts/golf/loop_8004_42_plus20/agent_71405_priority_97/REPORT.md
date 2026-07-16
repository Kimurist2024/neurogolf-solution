# 71405 priority six — deep audit

## Outcome

All six new-pool models are cheaper under competition `score_and_verify`, but
none is fixed or LB-white. Two remain isolated `LB_PROBE_REQUIRED` candidates:

1. task046 cost631→627 (`+0.006359`), SHA `fb649383...`.
2. task066 cost677→562 (`+0.186169`), SHA `bb8cebc...`, high risk.

The other four are rejected. No ZIP was created or merged. The only authority
was `submission.zip` SHA-256
`9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`.

## Probe candidates

### task046

- Competition cost: 631→627.
- Known: 267/267 in DISABLE_ALL/default at threads1/4.
- Fresh: 500/500 on each of two seeds, then 5000/5000 on a third seed, in both
  ORT modes.
- Checker, strict/data-propagating inference, static shapes, standard domains,
  no functions/banned ops, and Conv-family UB0 pass.
- Direct trace has zero declared/runtime shape mismatches, but one nonfinite
  intermediate value. Since nonfinite evidence is diagnostic rather than an
  LB oracle and this exact SHA has no history, it remains probe-only.

### task066

- Competition cost: 677→562.
- Known: 266/266 in all four configurations.
- Fresh: 470/500 and 478/500 on independent seeds, minimum94.0%, in both modes.
- Direct shape mismatches0 and UB0, but the graph contains a 61-input Einsum
  and one nonfinite intermediate.
- This exact SHA has no LB history. Three different task066 SHAs are already
  exact-LB-black (cost368/582/636), while the clean cost583 SHA is also still
  unprobed. The task history makes this a high-risk isolated probe, never a
  fixed adoption.

## Rejections

- task013 636→357: known masks and fresh1000/1000 agree, but known outputs
  contain 550,473 nonfinite values; direct trace exposes five shape cloaks,
  2,270 nonfinite intermediates, and max Einsum arity51. The current cost636
  SHA is LB-white, but this distinct cost357 SHA is not.
- task023 1622→1479: known×4 and truthful shapes pass, but fresh is 1/500 then
  0/500. This reproduces the known private-zero false-accept family.
- task069 541→524: disable-all is264/264, but default ORT cannot construct the
  session because `CenterCropPad` receives inconsistent shape/axes metadata.
- task044 1086→1076: known×4 passes and fresh reaches493/500 and495/500, but
  direct tracing finds two false 1×1×1×1 declarations for runtime
  1×10×30×30 tensors.

No candidate SHA has a prior exact-SHA text/LB verdict. The task066 black
history is explicitly recorded as related-task evidence, not as an exact-SHA
ban.

## Tail/order safety

Tasks044 and069 occur in the champion tail and would require in-place member
replacement with complete order preservation. Both were rejected before
packaging, so no tail member or archive order was changed. Tasks046 and066 are
not tail-order-sensitive.

## Evidence

- `audit/deep_audit.json`: profiles, static/UB checks, known×4, shapes, fresh,
  and exact-SHA searches.
- `audit/extra_fresh.json`: task013 reopened 2×500 and task046 5000-case run.
- `audit/task013_runtime_trace.json`: explicit task013 shape/nonfinite trace.
- `probe_manifest.json`: two isolated probe-only payloads.
- `winner_manifest.json`: empty fixed-winner list.
