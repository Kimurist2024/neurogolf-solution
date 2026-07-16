# 8006.61 isolated LB probe queue

Authority snapshot: `submission_base_8006.61.zip`, SHA-256
`9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118`,
MD5 `c90b23f514fe47c36f1d032bd0924662`.

Every ZIP has exactly 400 members, preserves authority member order, passes
CRC validation, and differs from the authority in exactly the named task.
None is fixed or LB-white until measured in isolation.

## Suggested probe order

1. `p00_task192_sound_cost1307.zip`: `1609 -> 1307`, projected `+0.207878`.
   This is the only queued candidate with a complete mathematical true-rule
   proof: 10,000/10,000 fresh, exhaustive 163-case sign proof, truthful trace,
   no UB, and at most 12 inputs per Einsum.
2. `p07_task046_cost627.zip`: `631 -> 627`, projected `+0.006359`,
   fresh 6000/6000 in both modes.  Shape-truthful and UB-clean; one traced
   intermediate nonfinite keeps it probe-only.
3. `p02_task396_cost1017.zip`: `1019 -> 1017`, projected `+0.001965`,
   minimum fresh rate 98.4%.  Safest task396 fallback in this lane.
4. `p03_task205_cost1038.zip`: `1042 -> 1038`, projected `+0.003846`,
   minimum fresh rate 97.4%.
5. `p05_task009_cost2616.zip`: `2619 -> 2616`, projected `+0.001146`,
   minimum fresh rate 95.6%.
6. `p09_task310_cost501.zip`: `566 -> 501`, projected `+0.121988`,
   known-complete and truthful, but one failure in 10,000 fresh cases plus six
   TfIdfVectorizer nodes and a 31-input Einsum make it a high-risk probe.
7. `p01_task066_cost583.zip`: `677 -> 583`, projected `+0.149484`,
   minimum fresh rate 95.8%.  High-value but high-risk giant-Einsum lineage;
   three different cheaper task066 SHAs are already LB-black.
8. `p08_task066_cost562.zip`: `677 -> 562`, projected `+0.186169`,
   minimum fresh rate 94.0%.  Highest-value and highest-risk task066 probe;
   it uses a 61-input Einsum and has one nonfinite traced intermediate.
9. `p04_task396_cost961.zip`: `1019 -> 961`, projected `+0.058603`,
   minimum fresh rate 97.2%.  Higher-risk task396 version probe.

`p06_task205_cost1041.zip` is a 97.6%-fresh fallback for task205 and should be
measured only if the cost1038 SHA is black; its projected gain is `+0.000960`.

No exact candidate SHA in this queue is LB-white yet.  In particular, the
task192 proof makes `p00` the strongest isolated probe, but it still must not
be merged until its exact SHA is measured on LB.

task219 candidates were intentionally not packaged because their minimum fresh
rate is 87.2%, below the current 90% threshold.  Do not merge probe ZIPs before
their exact SHA has an LB-white result.
