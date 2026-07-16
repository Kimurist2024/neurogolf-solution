# 8008.14 isolated LB probe queue

Authority: `submission_base_8008.14.zip`, SHA-256
`50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6`,
MD5 `db4da5cc59186b26572a380725bc2fdf`.

Every ZIP has exactly 400 unique members, preserves authority order and archive
comment, passes CRC validation, has no Conv-family short-bias mine, and differs
from the LB-verified authority in exactly the named task.  The 37 LB-white
71405 improvements are therefore preserved in every probe.

None of these exact candidate SHAs is fixed or LB-white until measured in
isolation.

## Suggested probe order

1. `p00_task192_sound_cost1307.zip`: `1609 -> 1307`, projected `+0.207878`.
   Exact decoded true-rule model: known 4x265/265, fresh 10,000/10,000,
   truthful trace, UB0, and exhaustive 163-case sign proof.
2. `p06_task349_exact_cost3556.zip`: `3564 -> 3556`, projected `+0.002247`.
   All-input integer/table identity proof plus 20,000/20,000 raw equality
   across two seeds and both ORT modes.  This preserves the LB-white authority
   exactly instead of approximating its true rule.
3. `p02_task205_cost1038.zip`: `1042 -> 1038`, projected `+0.003846`,
   minimum fresh rate 97.4%.
4. `p03_task009_cost2616.zip`: `2619 -> 2616`, projected `+0.001146`,
   minimum fresh rate 95.6%.
5. `p01_task396_cost1017.zip`: `1019 -> 1017`, projected `+0.001965`,
   minimum fresh rate 98.4%.  Task396 is version-sensitive: the separate
   71405 cost960 SHA was LB-black, so this remains probe-only.
6. `p04_task396_cost961.zip`: `1019 -> 961`, projected `+0.058603`,
   minimum fresh rate 97.2%.  This is a different exact SHA from the known
   cost960 black payload, but the task lineage makes it high risk.

`p05_task205_cost1041.zip` is a 97.6%-fresh fallback and should be measured
only if task205 cost1038 is black; projected gain is `+0.000960`.

The old `lb_probe_queue_8006_61/` is historical evidence only.  Submitting any
ZIP from it would revert the 37 LB-white 71405 improvements.
