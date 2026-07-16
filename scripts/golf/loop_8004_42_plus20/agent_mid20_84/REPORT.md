# mid20_84 — 20 additional target files

## Result

No safe score improvement was found. Projected accepted gain is `0.0`; the
winner manifest is empty. No submission ZIP, score ledger, CSV, or root score
pointer was modified.

The sole authority for this lane is `submission_base_8005.17.zip`, SHA-256
`c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`.
The 20 targets were tasks 008, 014, 037, 062, 092, 099, 109, 112, 160, 168,
245, 250, 275, 279, 297, 345, 374, 394, 397, and 398.

## Exhaustive repository screen

All loose ONNX files and every task member of every repository ZIP were
inventoried and SHA-deduplicated. The scan found 878 non-authority model SHAs.
The authority-corrected run submitted 90 models to actual runtime profiling
and 28 to complete known-example scoring. None reached the safe fresh gate.

Terminal stages across all 878 models:

- static cost reject: 570
- policy reject: 129
- structural reject: 89
- actual runtime-cost reject: 62
- complete-known reject: 28

The old helper cost table was not used as final authority: it understated
task014 as 203 instead of 370, task109 as 273 instead of 405, and task245 as
291 instead of 387. All 20 tasks were rescanned against costs produced by
`score_and_verify` from the exact ZIP members.

## task297 quarantine

The only lower-cost known/fresh-correct model is the zero-kernel trim at cost
361 versus 371. It passed 265/265 known examples in both ORT modes, the initial
500-case dual-ORT gate, and two independent 5000-case dual-ORT confirmations
with zero errors.

It is still rejected: its Conv uses `pads=[0,0,0,-24]`. Negative Conv padding
is outside the non-negative ONNX schema contract, so ORT's deterministic crop
extension is not accepted under fail-closed structural policy. The two tested
schema-compliant forms cost 484 (Slice) and 511 (Split/re-concat), both above
the authority cost.

## Additional exact-reduction audit

The exact authority members were independently checked for unused
initializers, bit-identical duplicate initializers, removable zero Conv-family
biases, and bypassable Identity nodes. These transformations produced zero
candidate models across the 20 targets. Tasks112 and 168 remained fail-closed
because of documented private-zero/unsound history and lack of a complete
true-rule proof.

## Evidence

- `result.json`: final decision and corrected authority costs
- `winner_manifest.json`: empty promotion manifest
- `authority_rescan/rescreen.json`: all 878 candidate rows
- `audit/inventory_summary.json`: per-task inventory and terminal stages
- `audit/task297_schema_rejection.json`: full task297 rejection evidence
- `authority_official_profiles.json`: exact ZIP authority profiles
- `mechanical_reductions.json`: exact-reduction opportunity audit
