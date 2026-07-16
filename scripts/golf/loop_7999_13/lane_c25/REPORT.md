# C25 — task131 / task251 strict-safety optimization report

## Outcome

No model was promoted. The exact `7999.13` ZIP and every root score/artifact
file remain byte-identical, so this lane's score gain is `+0.00` and it
introduces zero new error tasks.

Both assigned exact members pass all 266 known examples under the scorer's
execution path, but neither passes the required strict runtime-shape gate:

| task | exact cost | known, ORT disabled | known, ORT default | runtime shape mismatches |
|---:|---:|---:|---:|---:|
| 131 | 746 | 266/266, errors 0 | 266/266, errors 0 | 16 |
| 251 | 755 | 266/266, errors 0 | session creation fails | 59 |

The task251 default-mode failure is a `CenterCropPad` shape-arity error. These
members were treated only as immutable baselines, not as safe structures from
which to derive a promotion.

## Generator truth

The readable, specification-derived references were reverified independently:

- task131: move the green object adjacent to the red line and place the cyan
  separator one cell beyond it; visible `266/266`, fresh seeds `0..4999`
  `5000/5000`, errors 0.
- task251: flood-fill boundary-connected black cells and paint only enclosed
  black cells blue; visible `266/266`, fresh seeds `0..4999` `5000/5000`,
  errors 0.

Evidence: `reference_verification.json`.

## Full retained-archive audit

The shared archive inventory had already scanned 1,196 ZIPs, 448,568 ZIP
members, 233,751 loose files, and 13,591 unique non-baseline models. C25
reaudited every retained promising lineage for these tasks: five for task131
and four for task251.

### task131

| lineage | actual result | strict rejection |
|---|---|---|
| archive r01 | cost 596; both ORT modes 266/266 | 11 shape mismatches, `TfIdfVectorizer` lookup red flag; historical dual-ORT fresh5000 only 782/5000 with 4,218 failures |
| archive r02 | cost 627; both modes 266/266 | 11 shape mismatches and lookup red flag |
| archive r03 | cost 701; both modes 266/266 | 10 shape mismatches and lookup red flag |
| archive r04 | no executable score | ORT has no implementation for its `TopK(11)` type signature |
| archive r05 | no executable score | same `TopK(11)` failure |

The known-correct, runtime-shape-truthful control costs 24,927, well above the
746 exact baseline. No truthful algebraic shave was available: all 128 exact
initializer elements are referenced.

### task251

| lineage | actual result | strict rejection |
|---|---|---|
| archive r01 | cost 763 | not cheaper; 63 shape mismatches; default ORT session fails |
| archive r02 | cost 760 | not cheaper; 64 shape mismatches; default ORT session fails |
| archive r03 | cost 582; disabled-mode 266/266 | 58 shape mismatches; default ORT session fails |
| archive r04 | cost 709; disabled-mode 266/266 | 64 shape mismatches; default ORT session fails |

The truthful, both-ORT known-correct control costs 24,708, above the 755 exact
baseline. The prior nine-template QLinearConv rebuild costs 1,869 and still
contains 30 declared/runtime shape mismatches, so it is neither cheaper nor a
strictly truthful finalist. Historical 1030/1031/1032/1059 lineages also fail
the clipped-rectangle adversarial seed 313630.

The archive inventory and historical failures are consolidated in
`history_audit.json`; all model-level costs, hashes, operators, checker results,
dual-ORT outcomes, shape traces, and bias checks are in `model_audit.json`.

## Promotion gate and root integrity

There is no finalist because no model is simultaneously strictly cheaper,
known-correct in both ORT modes, and runtime-shape truthful. Consequently the
external validator is marked not applicable rather than being run on an
already-rejected model.

`root_integrity.json` confirms the starting hashes are unchanged for:

- `submission_base_7999.13.zip`
- `all_scores.csv`
- `best_score.json`
- all 402 files under `artifacts/handcrafted`

All C25 implementation and review work was performed directly by Codex. Kimi
was not invoked.

## Reproduction

```bash
.venv/bin/python scripts/golf/loop_7999_13/lane_c25/dump_models.py
.venv/bin/python scripts/golf/loop_7999_13/lane_c25/audit_models.py
.venv/bin/python scripts/golf/loop_7999_13/lane_c25/verify_references.py
.venv/bin/python scripts/golf/loop_7999_13/lane_c25/finalize_evidence.py
```
