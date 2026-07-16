# History miner — 8004.50 rebase

## Outcome

No historical artifact is admissible on top of `submission_base_8004.50.zip`.
The accepted set is empty and the projected gain from this lane is `+0.0`.

The scan re-used the exhaustive all-400 inventory (1,196 ZIPs, 448,568 ZIP
members, 233,751 loose ONNX observations, 13,591 unique nonbaseline SHAs), then
screened 1,882 later files / 643 later SHAs and re-read 134 final acceptance or
comparison JSON files. The 22 structured reports under `others/` produced no
additional final candidate.

## Decisive rechecks

- `task204` looked like a large static win, but current runtime profiling gives
  `2544` against current `2240`; the false static floor `573` came from small
  value-info declarations. It is a regression and a shape-cloak-style lead.
- `task023` is truly cheaper (`1622 -> 1497`) and known-complete, but an
  independent generator screen scored only `2/500` in each ORT mode (`0.4%`).
- `task202`, `task205`, and `task344` pass the user's numerical 95% fresh bar,
  but are excluded by the explicit private-zero policy. task202/task344 also
  change 21/24-input floating Einsum contractions; task205 is worse than its
  current member on fresh cases and emits arbitrary-input shape warnings.
- `task153` is `5000/5000` fresh but retains the known QLinearConv short-bias UB
  (9 bias elements for 10 output channels), so it remains forbidden.

Machine-readable details, including path, SHA, current/candidate cost, gain,
known/fresh/error/strict/UB evidence and lineage, are in
`candidate_manifest.json`. `history_inventory.json` contains the deduplicated
final-acceptance record inventory. No submission ZIP or protected root file was
modified.
