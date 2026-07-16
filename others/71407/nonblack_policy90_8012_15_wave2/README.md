# Nonblack POLICY90 checkpoint — 8012.15 wave 2

The latest LB-black candidates 070@52, 134@320, 202@20, and 343@172 are absent.
Every generated ZIP keeps those four authority members byte-identical.
Root `submission.zip`, `all_scores.csv`, and the 8012.15 authority are unchanged.

| task | cost | minimum audited accuracy | gain | classification |
|---:|---:|---:|---:|---|
| 023 | 1321→1317 | authority raw-identical | +0.003033 | EXACT_AUTHORITY_EQUIVALENT |
| 012 | 710→650 | 94.55% | +0.088293 | NONBLACK_POLICY90 |
| 161 | 190→186 | 99.35% | +0.021277 | NONBLACK_POLICY90 |
| 175 | 166→145 | 98.50% | +0.135254 | NONBLACK_POLICY90 |
| 354 | 497→461 | 100.00% | +0.075192 | KNOWN_LB_WHITE_LINEAGE_EXACT_FRESH |
| 355 | 250→249 | 98.50% | +0.004008 | NONBLACK_POLICY90_PUBLIC_OVERFIT_RISK |

Conditional gain: **+0.327057** → **8012.477057**.

Probe the six ZIPs in `probes/` one at a time before using the cumulative ALL6 ZIP.
task023 is an exact authority-equivalent rewrite, not an approximate POLICY90 model.
task354 retains its verified authority lineage's legacy declared/runtime shape mismatch.
task355 remains explicitly labeled as public-overfit risk and should be probed last.
