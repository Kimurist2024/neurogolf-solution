# task367 / task382 / task270 SOUND strict-lower audit

## Verdict

No candidate is admissible. `winner_manifest.json` is intentionally empty, projected gain is `0.0`, and no promotion was performed.

The immutable authority was `submission_base_8009.46.zip` / `submission.zip`, both SHA-256 `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927` at final verification. The root submission, score files, `all_scores.csv`, `others/`, and `docs/` were not changed.

## Authority audit

| task | authority cost | known dual ORT | runtime-shape truth | decisive SOUND issue |
|---:|---:|---|---|---|
| 367 | 2179 | optimizer disabled: 266/266 at 1 and 4 threads; default: 0/266 | 65 mismatches; declared output `[1,1,11,1]`, actual `[1,10,30,30]` | shape cloak and default-optimizer disagreement |
| 382 | 820 | optimizer disabled: 254/266; default does not execute correctly | 19 mismatches; declared output `[1,10,1,1]`, actual `[1,10,30,30]` | authority is already incorrect, shape-cloaked, and default ORT fails |
| 270 | 587 | 266/266 in all four configurations | 4 intermediate-shape mismatches | final 19-input `Einsum` is a giant contraction, so it is policy-rejected despite known correctness |

All three authority members pass the static checker, strict data propagation, finite-initializer, standard-domain, and official Conv-bias UB0 checks. Those static checks do not override the runtime-shape and dual-ORT failures above.

## Search and exact shaves

- Initializer inspection found no unused initializer, byte-identical initializer alias, dead node, duplicate node, or removable optional output in any target.
- ONNX normalization and conservative optimizer sweeps produced no strict-lower profile. task367 and task382 each had one byte-distinct optimizer result at exactly the authority cost; task270 was unchanged.
- Exact `CastLike(x, ref) -> Cast(x, to=dtype(ref))` substitutions were tested with the now-unused reference initializer removed. The task367 results cost 10663, 2216, and 10700; the task270 result cost 626. None was lower or runtime-shape truthful.
- The current-authority rescreen retained 11 byte-distinct archive-frontier models: 5 for task367, 5 for task382, and 1 for task270. No model survived. task367 costs were 2247..2309; task270 cost 595. task382 had costs 808..818, but four were officially incorrect and the cost-814 model was a known shape cloak/default-ORT failure.

## History coverage

- task367: 901 historical source references collapsed to 69 unique SHAs in the prior headroom scan; none beat the current 2179 authority. The independently rebuilt truthful exact control is SHA `7673a580bc645f491eb85b110b142d3c6ed5dcac91df0b676c9556c6b156bdbf`, cost 3913, and therefore not lower.
- task382: all 62 unique SHAs in the exhaustive prior rescreen were accounted for: 34 static rejects, 11 actual-cost rejects, 9 known rejects, 7 structure rejects, and 1 dual-ORT reject. Survivors: 0.
- task270: the current loose-history inventory covered 600 observations and 38 unique nonbaseline SHAs. It already had zero numeric model below cost 594; the present authority is still lower at 587. The retained cost-595 exact candidate is also a giant-contraction model.
- The shared all-400 inventory covered 1,196 ZIPs, 448,568 ZIP members, 233,751 loose observations, and 13,591 unique nonauthority SHAs before its static/structure frontier reduction.

Exact counts and source manifests are in `history_coverage.json` and `archive_rescreen.json`.

## True-rule basis

- task367: fill black interiors of hollow gray rectangles yellow while preserving gray borders and connector lines, including one-column edge clipping.
- task382: extend cyan boundary seeds and shift them perpendicular to the boundary by the cumulative number of passed red markers over all eight flip/gravity orientations.
- task270: preserve both flower centers and move every present remote petal inward to its adjacent cardinal cell around the corresponding center.

The generator paths, SHA-256 hashes, and prior independent proof reports are pinned in `true_rule_evidence.json`.

## Fresh policy

New fresh validation was not run. The required ordering was strict-lower -> full checker -> strict data propagation -> runtime-shape truth -> known dual ORT with zero runtime errors -> policy exclusions -> two-seed fresh. The structural-gate survivor count was zero, so running fresh on rejected candidates would not create an admissible winner.

## Outputs

- `result.json`: terminal no-winner result
- `winner_manifest.json`: empty winner manifest
- `manifest.json`: authority and evidence index
- `authority_audit.json`: full checker, shape trace, graph inventory, and four-config known audit
- `initializer_analysis.json`, `exact_scan.json`, `optimizer_sweep.json`, `castlike_shave_scan.json`: exact-shave evidence
- `history_coverage.json`, `archive_rescreen.json`, `true_rule_evidence.json`: history and semantic evidence

Final integrity checks: every JSON file parsed successfully, every audit script compiled, `submission.zip == submission_base_8009.46.zip`, and `all_scores.csv` remained SHA-256 `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`.
