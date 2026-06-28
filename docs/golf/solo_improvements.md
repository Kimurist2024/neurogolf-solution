# Solo deep-dive improvements log

One row per task the sequential campaign closed. cost_before is the
6713.69 champion cost; cost_after is the fresh-gated adopted cost.

| task | status | cost_before | cost_after | score_before | score_after | Δscore | rounds | fresh_k | provenance |
|---|---|---|---|---|---|---|---|---|---|
| 234 | PROMOTED | 19943 | 7359 | 15.099 | 16.096 | +0.997 | 1 | 500 | Provenance |
| 044 | FLOOR | 21617 | 21617 | 15.019 | 15.019 | +0.000 | 3 | 500 | Provenance |
| 308 | PROMOTED | 21799 | 7435 | 15.010 | 16.086 | +1.076 | 1 | 500 | Provenance: fully spec-derived from `inputs/arc-gen-repo/tasks/task_c8cbb738.py`. |
| 145 | PROMOTED | 22230 | 17659 | 14.991 | 15.221 | +0.230 | 1 | 500 | Provenance declaration |
| 324 | PROMOTED | 22579 | 16550 | 14.975 | 15.286 | +0.311 | 1 | 500 | Provenance declaration |
| 378 | PROMOTED | 24148 | 15911 | 14.908 | 15.325 | +0.417 | 1 | 500 | Provenance |
| 071 | PROMOTED | 24170 | 7780 | 14.907 | 16.041 | +1.134 | 1 | 500 | Provenance declaration |
| 340 | PROMOTED | 24505 | 10743 | 14.893 | 15.718 | +0.825 | 1 | 500 | Provenance |
| 170 | PROMOTED | 24799 | 9078 | 14.881 | 15.886 | +1.005 | 1 | 500 | Provenance |
| 222 | PROMOTED | 24819 | 13875 | 14.881 | 15.462 | +0.582 | 1 | 500 | Provenance |
| 279 | FLOOR | 25042 | 25042 | 14.872 | 14.872 | +0.000 | 3 | 500 | (no REPORT.md) |
| 383 | FLOOR | 25394 | 25394 | 14.858 | 14.858 | +0.000 | 3 | 500 | Provenance: SPEC-DERIVED (fully compiled from the generator) |
| 004 | FLOOR | 25423 | 25423 | 14.857 | 14.857 | +0.000 | 3 | 500 | (no REPORT.md) |
| 255 | FLOOR | 26693 | 26693 | 14.808 | 14.808 | +0.000 | 3 | 500 | Provenance |
| 396 | FLOOR | 26990 | 26990 | 14.797 | 14.797 | +0.000 | 3 | 500 | Provenance: SPEC-DERIVED (fully compiled from the generator) |
| 080 | FLOOR | 27843 | 27843 | 14.766 | 14.766 | +0.000 | 3 | 500 | Provenance |
| 161 | FLOOR | 28822 | 28822 | 14.731 | 14.731 | +0.000 | 3 | 500 | (no REPORT.md) |
| 009 | FLOOR | 29242 | 29242 | 14.717 | 14.717 | +0.000 | 3 | 500 | (no REPORT.md) |
| 182 | FLOOR | 29723 | 29723 | 14.700 | 14.700 | +0.000 | 3 | 500 | (no REPORT.md) |
| 319 | FLOOR | 30518 | 30518 | 14.674 | 14.674 | +0.000 | 3 | 500 | (no REPORT.md) |
| 063 | FLOOR | 30732 | 30732 | 14.667 | 14.667 | +0.000 | 3 | 500 | (no REPORT.md) |
| 328 | FLOOR | 30736 | 30736 | 14.667 | 14.667 | +0.000 | 3 | 500 | (no REPORT.md) |
| 064 | FLOOR | 30889 | 30889 | 14.662 | 14.662 | +0.000 | 3 | 500 | Provenance declaration |
| 377 | FLOOR | 30939 | 30939 | 14.660 | 14.660 | +0.000 | 3 | 500 | (no REPORT.md) |
| 216 | FLOOR | 31399 | 31399 | 14.645 | 14.645 | +0.000 | 3 | 500 | Provenance declaration |
| 399 | MERGE | 1184 | 236 | 17.923 | 19.536 | +1.613 | - | 500 | handcrafted lane (fresh-gated) |
| 003 | MERGE | 1113 | 377 | 17.985 | 19.068 | +1.083 | - | 500 | handcrafted lane (fresh-gated) |
| 123 | MERGE | 4103 | 1456 | 16.681 | 17.717 | +1.036 | - | 500 | handcrafted lane (fresh-gated) |
| 235 | MERGE | 1230 | 439 | 17.885 | 18.916 | +1.030 | - | 500 | handcrafted lane (fresh-gated) |
| 181 | MERGE | 1194 | 453 | 17.915 | 18.884 | +0.969 | - | 500 | handcrafted lane (fresh-gated) |
| 180 | MERGE | 1344 | 594 | 17.797 | 18.613 | +0.817 | - | 500 | handcrafted lane (fresh-gated) |
| 072 | MERGE | 1309 | 611 | 17.823 | 18.585 | +0.762 | - | 500 | handcrafted lane (fresh-gated) |
| 052 | MERGE | 1120 | 592 | 17.979 | 18.616 | +0.638 | - | 500 | handcrafted lane (fresh-gated) |
| 257 | MERGE | 1077 | 605 | 18.018 | 18.595 | +0.577 | - | 500 | handcrafted lane (fresh-gated) |
| 267 | MERGE | 1207 | 772 | 17.904 | 18.351 | +0.447 | - | 500 | handcrafted lane (fresh-gated) |
| 012 | MERGE | 1220 | 820 | 17.893 | 18.291 | +0.397 | - | 500 | handcrafted lane (fresh-gated) |
| 115 | MERGE | 4119 | 2930 | 16.677 | 17.017 | +0.341 | - | 500 | handcrafted lane (fresh-gated) |
| 039 | MERGE | 1039 | 797 | 18.054 | 18.319 | +0.265 | - | 500 | handcrafted lane (fresh-gated) |
| 114 | MERGE | 1249 | 1035 | 17.870 | 18.058 | +0.188 | - | 500 | handcrafted lane (fresh-gated) |
| 269 | MERGE | 1138 | 951 | 17.963 | 18.142 | +0.180 | - | 500 | handcrafted lane (fresh-gated) |
| 106 | MERGE | 1164 | 996 | 17.940 | 18.096 | +0.156 | - | 500 | handcrafted lane (fresh-gated) |
| 101 | MERGE | 90150 | 86141 | 13.591 | 13.636 | +0.045 | - | 500 | handcrafted lane (fresh-gated) |
| 292 | MERGE | 1348 | 1306 | 17.794 | 17.825 | +0.032 | - | 500 | handcrafted lane (fresh-gated) |
| 043 | MERGE | 1105 | 1092 | 17.992 | 18.004 | +0.012 | - | 500 | handcrafted lane (fresh-gated) |
| 060 | MERGE | 1136 | 1134 | 17.965 | 17.966 | +0.002 | - | 500 | handcrafted lane (fresh-gated) |
| 367 | MERGE-EXT | 109037 | 13670 | 13.401 | 15.477 | +2.076 | - | ext | provided submission.zip (fresh 0/500) |
| 279 | PROMOTED | 25042 | 13732 | 14.872 | 15.473 | +0.601 | 2 | 500 | (no REPORT.md) |
| 383 | PROMOTED | 25394 | 15180 | 14.858 | 15.372 | +0.515 | 1 | 500 | Provenance |
| 004 | PROMOTED | 25423 | 12159 | 14.857 | 15.594 | +0.738 | 1 | 500 | Provenance |
| 255 | FLOOR | 26693 | 26693 | 14.808 | 14.808 | +0.000 | 3 | 500 | Provenance |
| 255 | AB-DROPPED | 26693 | 9641 | 14.808 | 15.826 | +1.018 | - | 24/500 | A/B (<= 5% fresh-fail)  |
| 396 | PROMOTED | 26990 | 16457 | 14.797 | 15.291 | +0.495 | 2 | 500 | Provenance: SPEC-DERIVED |
| 080 | PROMOTED | 27843 | 13881 | 14.766 | 15.462 | +0.696 | 1 | 500 | Provenance |
| 161 | PROMOTED | 28822 | 16615 | 14.731 | 15.282 | +0.551 | 1 | 500 | Provenance |
| 009 | PROMOTED | 29242 | 8608 | 14.717 | 15.940 | +1.223 | 1 | 500 | Provenance |
| 182 | PROMOTED | 29723 | 17782 | 14.700 | 15.214 | +0.514 | 1 | 500 | Provenance |
| 281 | PROMOTED | 2006 | 1778 | 17.396 | 17.517 | +0.121 | 1 | 30 | Provenance declaration |
| 157 | PROMOTED | 2235 | 1984 | 17.288 | 17.407 | +0.119 | 6 | 30 | Provenance Declaration |
| 250 | PROMOTED | 2389 | 1989 | 17.221 | 17.405 | +0.183 | 1 | 30 | Provenance |
| 096 | PROMOTED | 2450 | 1854 | 17.196 | 17.475 | +0.279 | 1 | 30 | Provenance Declaration |
| 029 | FLOOR | 2540 | 2390 | 17.160 | 17.221 | +0.061 | 8 | 30 | Provenance |
| 005 | FLOOR | 2545 | 2534 | 17.158 | 17.162 | +0.004 | 7 | 30 | Provenance declaration |
