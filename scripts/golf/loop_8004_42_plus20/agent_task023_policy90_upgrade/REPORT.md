# task023 POLICY90 upgrade — final rejection

## Decision

`winner = null`. No candidate from this lane is eligible for adoption.

- Immutable authority: `submission_base_8009.46.zip`
- Authority ZIP SHA-256: `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- Authority task023 SHA-256: `bd242d29ab9514b2432dce31e6df28dd67f00bf1bdcb54c8a00f28614f974fb0`
- Authority task023 cost: `1622`
- Required policy: known-complete, error-free, truthful/static, cost `<1622`, and at least 90% on two disjoint fresh streams
- Best independently measured existing lower-cost variant in this analysis: `44512/50000 = 89.024%`
- Decision: `REJECT_POLICY90`

No root, stage, artifact, or submission archive was changed. No candidate was
created in this lane.

## Generator and failure diagnosis

`inputs/arc-gen-repo/tasks/task_150deff5.py` generates nonoverlapping gray
rectangles: 2x2 boxes and 1x3/3x1 sticks. Boxes become cyan; sticks become red.
There are exactly two boxes at width 9 and three boxes at widths 10/11. The
gray input is not always uniquely decomposable into those objects when adjacent
rectangles form additional filled 2x2 windows.

The clean cost-1541 graph scores the 36 possible anchors with one clipped
uint8 6x6 `QLinearConv`, selects three anchors using `TopK`, and gates the third
selection off for width 9. Its main failure is geometric: adjacent sticks and
boxes create false filled 2x2 anchors that a single clipped linear score cannot
reliably separate from true boxes. The problem is much harder at widths 10/11,
where all three true boxes must outrank every false anchor.

On the fixed disjoint 50,000-case stream beginning at seed `230900001`, exact
QLinearConv saturation and TopK tie-order simulation gave:

| model | SHA-256 | cost | exact fresh |
|---|---|---:|---:|
| archive r03 | `9a2b7813889112837080e0364d0a8971671ca384079e188f460dee6b158b3ab1` | 1541 | 44022/50000 (88.044%) |
| root coordinate2 | `763c7002607625de9812bab2ea0cd6db73799a3dc7c8153eaf6c4ee7c7a1d346` | 1541 | 44512/50000 (89.024%) |
| rank82 integer root-c2 | `d9c9c5d471b34b6c35ffc6006d038b6e21ef3e91ae31145e85da8ca46ffff0e9` | 1541 | 44507/50000 (89.014%) |

For coordinate2, the width breakdown was 8062/8295 and 8165/8407 for the two
width-9/height regimes (97.19% combined), but only 28285/33298 across widths
10/11 (84.95%). Its accuracy by number of false filled 2x2 anchors was 99.84%
with none, 95.47% with one, 89.37% with two, 82.02% with three, and 74.40% with
four. This is the observed saturation mechanism, not a runtime error.

The earlier 448/500 (89.6%) sample is therefore below policy and is also not a
stable estimate: prior independent 5,000-case measurements were 88.06% and
88.20%, while the larger disjoint stream here is 88.044% for the same r03 SHA.

## Variants evaluated or designed

The three saved 1541-cost kernels above cover the archive kernel, a
known-preserving coordinate-tuned kernel, and the later known-preserving global
integer-coordinate refinement. Existing structured rank and integer searches
plateaued near 89%; changing only the 36 kernel bytes did not cross POLICY90.

A compact nonlinear morphology variant was costed but not implemented:

1. Replace the 6x6 one-output `QLinearConv` with a 4x4 two-output
   `QLinearConv`, shape `[1,2,6,6]`, pads `[1,1,0,0]`.
2. Combine the two clipped features with a 1x1 `QLinearConv` to recover
   `[1,1,6,6]` before the unchanged Flatten/TopK tail.
3. The first layer has 32 weights and the second has 2, versus 36 current
   weights. The extra activation memory is 72 bytes. Therefore its theoretical
   official-like cost is `1541 - 2 + 72 = 1611` without biases, or 1614 with
   valid bias vectors of lengths two and one.

This design fits below 1622, but it has no generated ONNX file and no measured
holdout result. It is not a candidate and must not be inferred to satisfy the
90% gate.

## Delegation record

The configured `~/.Codex/bin/ask-kimi` wrapper was absent. A direct Kimi launch
then failed before session creation because it could not create its home-session
directory under the sandbox; the later escalation was aborted. Kimi never
started a working session, edited files, generated a model, or ran validation.
All results in this report are the Codex-side read-only analysis completed before
the final stop instruction.

## Final review

The existing 1541 family remains structurally clean and known-complete in its
prior audits, but its fresh generalization is below POLICY90. Since no new
strictly-lower candidate with two qualifying holdouts was saved, adoption would
violate the requested gate. The authority must remain unchanged.
