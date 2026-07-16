# task002 / task012 / task107 strict-lower SOUND audit

## Verdict

No candidate is admissible. Winner count is `0`, projected gain is `0.0`, and no promotion was performed.

The immutable authority is `submission_base_8009.46.zip` / `submission.zip`, SHA-256 `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`. This lane did not modify the root submission, score files, `all_scores.csv`, `others/`, or `docs/`.

## Authority characterization

| task | competition actual profile | full/strict/UB0 | runtime shape | four-config known | SOUND disposition |
|---:|---:|---|---|---|---|
| 002 | 360 memory + 926 params = 1286 | pass | truthful, 0 mismatches, finite | 268/268 raw-equal in disabled/default x threads 1/4 | reject: final `Einsum` has 66 inputs; generator is not an input-only function |
| 012 | 0 memory + 710 params = 710 | pass | truthful, 0 mismatches, finite | 265/265 raw-equal in all four configurations | SOUND authority; no lower exact model |
| 107 | 365 memory + 299 params = 664 | pass | 14 mismatches and 904 nonfinite values on trace witness | threshold-correct 266/266, but each authority copy emits 1,061,773 nonfinite values over the known corpus | reject: 58-input `Einsum`, shape cloak, nonfinite outputs |

All Conv-bias UB findings are empty. Passing the competition known threshold is not treated as permission to admit giant contractions, NaN/Inf behavior, false shape declarations, lookup, or private-zero mechanisms.

## History inventory and strict-lower frontier

### task002

The headroom inventory collapsed 877 historical references to 39 unique SHAs: 28 cost-floor rejects, 10 structural rejects, and the authority member. There is no strict-lower frontier.

The generator itself contains latent pot parameters that are not recoverable from the input. Two positive-probability legal parameterizations have a byte-identical input but different required output. Therefore no deterministic single-input ONNX model can have a full-support correctness guarantee; a behavior-changing cheaper candidate is prohibited independently of cost.

### task012

The earlier inventory covered 871 references / 20 unique SHAs; a later loose scan covered 586 observations / 16 nonbaseline SHAs. The only numeric frontier is the cost-500 Conv family.

The current-authority re-audit of SHA `a3640a15252636a0bc7e3ed3e56353e3d1693c09d3fcadaf4a899c9490531894` found:

- competition profile: cost 500, officially incorrect;
- full/strict and runtime-shape truth: pass;
- disabled/default x threads 1/4: 235/265 correct in every configuration;
- runtime errors and nonfinite values: 0;
- raw equality to authority: 0/265.

The other two recorded cost-500 serializations were already officially scored incorrect in the headroom scan. The complete prior finite-domain proof checked all 392 generator parameter states and all 1,712 sub-710 one-node Conv geometries; none is exact. The authority already has zero charged intermediate memory, and its 700 weights plus 10 required biases form the current exact floor.

### task107

The headroom inventory covered 899 references / 73 unique SHAs. The later exhaustive rescreen covered 77 unique SHAs: 59 static-cost rejects and 18 structure rejects. The only retained frontier below 664 is SHA `a48fb6a68e06e981ff03d76bfc07fe68efa057ad51712acd0506a6cf10828ee2` at cost 638.

Its current-authority re-audit found:

- competition known threshold: 266/266 in disabled/default x threads 1/4;
- runtime/session errors: 0;
- raw authority equality: 0/266 in every configuration;
- runtime-shape mismatches: 13;
- nonfinite values: present in every configuration;
- final `Einsum`: 66 inputs.

It is a terminal SOUND rejection even though its threshold output reproduces the known corpus. The prior exact/fresh factorization family is also based on 50..68-input monolithic contractions and inherited false declarations, so historical fresh success does not waive this lane's structural policy.

## Current exact shave audit

- No target has unused initializers, byte-identical initializer aliases, CastLike-only type-reference initializers, or explicit default-input holes.
- Conservative dead-node, initializer, optional-output, no-op, CSE, constant-fold, and normalization passes produced no strict-lower model.
- Conservative ORT optimizer sweeps produced no strict-lower model.
- Removing task012's inferable `kernel_shape` attribute is semantically exact but remains cost 710; serialized bytes are not scoring parameters.
- Removing task107's inferable Conv `kernel_shape` and five explicit default `Pad(mode="constant")` attributes remains cost 664 and preserves its 14 shape mismatches/nonfinite behavior.
- Replacing task107 value-info declarations with witnessed runtime shapes raises competition actual cost to 1286. Shape mismatches become zero, but nonfinite outputs and the 58-input giant contraction remain, so it is still not SOUND or lower.
- task012 weight/bias trimming and smaller direct-output geometries are closed by the prior finite-domain lower-bound search. task002 behavior-changing shaves cannot receive a full-support guarantee.

## Fresh decision

New fresh validation was not run. The mandated order is strict lower -> competition actual profile -> full checker/strict data propagation -> truthful runtime node shapes -> disabled/default x threads 1/4 known-all -> runtime and nonfinite zero -> raw authority equivalence or independently guaranteed true-rule behavior -> policy exclusions -> fresh. No candidate reached the final prerequisite gate.

## Evidence

- `authority_audit.json`: authority cost, full/strict structure, runtime node shapes, and four-config known audit
- `initializer_analysis.json`, `exact_scan.json`, `optimizer_sweep.json`, `defaults_and_shapes_scan.json`: current exact-shave evidence
- `history_frontier_audit.json`: current-authority full re-audit of cost500/cost638 frontiers
- `history_coverage.json`: historical SHA coverage and source hashes
- `true_rule_evidence.json`: generator hashes, rules, ambiguity proof, and finite-domain reports
- `result.json`, `manifest.json`, `winner_manifest.json`: terminal empty-winner result

Final integrity verification parsed every JSON artifact, compiled every lane script, confirmed the three protected brief files are byte-identical to `HEAD`, and confirmed `submission.zip == submission_base_8009.46.zip` at the authority SHA above. `all_scores.csv` remained SHA-256 `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`.
