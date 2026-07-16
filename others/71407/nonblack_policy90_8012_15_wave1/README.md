# Nonblack POLICY90 checkpoint — 8012.15 wave 1

The four latest LB-black candidates (070@52, 134@320, 202@20, 343@172) are not present.
Root `submission.zip`, `all_scores.csv`, and the 8012.15 authority were not modified.

| task | cost | minimum accuracy | gain | classification |
|---:|---:|---:|---:|---|
| 161 | 190→186 | 99.35% | +0.021277 | NONBLACK_POLICY90 |
| 175 | 166→145 | 98.50% | +0.135254 | NONBLACK_POLICY90 |
| 354 | 497→461 | 100.00% | +0.075192 | KNOWN_LB_WHITE_LINEAGE_EXACT_FRESH |
| 355 | 250→249 | 98.50% | +0.004008 | NONBLACK_POLICY90_PUBLIC_OVERFIT_RISK |

Conditional gain: **+0.235731** → **8012.385731**.

Use the four ZIPs in `probes/` individually first. The cumulative ALL4 ZIP is only a convenience and is not LB-guaranteed.
task354 is exact on the audited known/fresh sets and comes from a prior LB-white lineage, but retains its authority's legacy shape mismatch.
task355 is explicitly marked public-overfit risk.
