# True-rule rebuild lane — LB 8004.50

## Outcome

- Immutable comparison base: `submission_base_8004.50.zip`
  (`63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`).
- Nine unfixed high-cost/local-rule tasks were selected: 125, 145, 187, 192,
  196, 204, 208, 340, and 344.
- The 26 LB-fixed members were not edited or copied into a new ZIP.
- Safe accepted models: **0**. Projected gain: **+0.0**.
- No aggregate ZIP or root protected file was changed.

## Generator-rule evidence

The Sakana `p` rule for every selected task was executed independently on the
complete known corpus and 2,000 fresh generator cases. Tasks 125, 145, 192,
196, 204, 208, 340, and 344 are 100% on both sets. Task187 is complete-known
but only 1965/2000 fresh, proving that its supplied compact rule and the current
generator are not equivalent. The machine-readable evidence is
`true_rule_audit.json`.

## Only cheaper complete-known ONNX lead

`candidates/task192_true_local_lp_k4.onnx` uses an input-derived non-zero color
histogram and one bounded 3x4 Conv. It has no lookup table, no banned/nested op,
no giant Einsum, static truthful shapes, complete local and official known
correctness, margin minimum 0.999267578125, and cost 1322 versus 1609
(`+0.1964671266` projected).

It is rejected: independent seed 19299901 produced only 440/500 correct cases
(88.0%), with zero runtime errors. This is below the explicit 95% admission
floor. The model remains only as negative evidence and is absent from
`winner_manifest.json`.

## Structural rejections

| task | base cost | decisive blocker |
|---:|---:|---|
| 125 | 1050 | Reverse in-place scan makes identical original 3x3 patches map differently; exact unroll is costlier. |
| 145 | 5130 | Two flood fills plus connected-component extrema (Type C). |
| 187 | 1814 | Sakana rule itself disagrees with generator on 35/2000 fresh cases. |
| 192 | 1609 | Cheaper bounded classifier is only 88% independent-fresh. |
| 196 | 1210 | Closed/broken rectangles up to 5x5; truthful 5x5 dense Conv costs 2510. |
| 204 | 2240 | Horizontal context alone conflicts; required vertical+horizontal Conv exceeds base. |
| 208 | 1422 | Global maximum-empty-rectangle selection; local patches conflict. |
| 340 | 1173 | Whole-row/column color projection; local patches conflict. |
| 344 | 197 | Exact local mask floor is 900 bytes; dense 3x3 final Conv is 910 params. |

Experimental builders and failed candidates are retained for reproducibility,
but none is an admission candidate. `RESULTS.json` is authoritative.
