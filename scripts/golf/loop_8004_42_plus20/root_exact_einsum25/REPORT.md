# 8005.16 exact Einsum rescan

**Safe adoptees: 0.**  All 400 members of `submission_base_8005.16.zip`
were rescanned for parameter-reducing constant outer-product fusion and exact
sign absorption.

The only three structural candidates reproduce already classified unsafe
lineages:

- task048 outer fusions each remove one parameter, but are exactly equivalent
  to the private-risk incumbent whose independent fresh results were only
  90.06% and 91.08%, not the required private-lineage 100%. Both are rejected.
- task333 sign absorption removes two parameters but retains/changes a giant
  floating Einsum contraction and is excluded by the no-giant safety gate.

No new initializer alias or truthful metadata reduction was found. Gain
counted is `+0.0`. Evidence: `scan_report.json`.
