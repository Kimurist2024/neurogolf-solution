# Lane B21 — task226 / task338 strict Wave15 audit

## Result

No candidate is admissible. Accepted gain is **0.0**. The exact Wave15 members
are byte-identical to the pinned 7999.13 members. No root ZIP, CSV, score
pointer, or shared handcrafted artifact was modified.

## task226 — generator-complete constant-code search

The exact member is SHA-256
`342ff4b0df090df3cb1fdea435049e05f9e317f4775af82a14ded63b2a490c13`,
cost **399 = 346 memory + 53 parameters**. It passes full checking, strict
shape/data propagation, has no runtime shape mismatch, and is **133/133** on
complete known data in each ORT mode with zero errors.

This lane compiled the finite generator domain rather than sampling it. There
are 136 valid `(wides, talls)` combinations. The current graph's row/column
program reduces those cases to 25 symbolic feature/label records feeding its
two-dimensional QLinearConv classifier.

Two exhaustive initializer-reuse searches were run:

- all 36 directed reuses of existing same-shape row/scalar code values, with
  the QLinearConv weights refit;
- arbitrary shared values, not restricted to incumbent bytes: four scalar
  pairs × 256 values and three two-byte row-vector pairs × 65,536 values.

The three column-code equalities and all three row-vector equalities create a
feature/label collision for every possible shared value. For `x_zp = R_B`,
108/256 values remain collision-free, but none admits the required five robust
homogeneous linear output classifiers. Consequently no model reaches the ORT
candidate gate. This also agrees with the prior full archive screen: its
lowest historical static floor is 400, above the exact cost 399.

Evidence:

- `task226_reuse_search.json` — existing-value directed search and full-domain
  baseline checks;
- `task226_arbitrary_ties.json` — arbitrary-value exhaustive symbolic search;
- `search_task226_reuse.py` and `search_task226_arbitrary_ties.py` — reproducible
  Codex implementations.

## task338 — inherited shape cloak blocks safe shaving

The exact member is SHA-256
`edcac049616e90e42b848d1a719b3af7a4a078b5d1180a3cdf0ecf60e340a01d`,
nominal cost **426 = 424 memory + 2 parameters**. ORT_DISABLE_ALL remains
267/267 on known data. It is not a strict-safe base, however:

- default ORT rejects session construction because the declared output
  `[1,10,1,1]` conflicts with the inferred `[1,10,30,30]`;
- exposing intermediates makes ORT fail on the inherited CenterCropPad/Cast
  buffer plan (`[1,1,29,30]` versus `[1,1,30,30]`).

Every known below-nominal alternative remains terminally rejected:

- static 334: a new unsafe 24-input giant Einsum;
- actual 424: fails known/runtime correctness;
- static 425 Cast rewrite: known-correct, but its truthful profiled cost is
  18,423;
- Boolean Min/Max fusion: deterministic Slice/buffer-reuse RuntimeException.

Thus a topology edit cannot qualify under the required no-shape-cloak and
both-ORT gates. The nominal 426 number cannot be safely shaved by preserving
the invalid allocation metadata, and removing that metadata exposes a cost
far above the incumbent.

## Admission gate

`candidate_manifest.json` records zero accepted models. Fresh 5,000 was not
started because no candidate survived the earlier mandatory gates: strictly
cheaper actual cost, complete known correctness in both ORT modes, runtime
error zero, and truthful static/runtime shapes.

Baseline identity, structure, runtime-shape trace, and dual-known evidence are
in `baseline_audit.json`; historical dispositions are independently recorded
by `lane_a10/history_screen.json`, `lane_a15/retained_scan.json`, and the prior
`lane_a5/winner_manifest.json`.
