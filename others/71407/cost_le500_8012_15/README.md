# cost<=500 optimization checkpoint (8012.15 authority)

> **WITHDRAWN / 使用禁止 (2026-07-15):** この旧チェックポイントの4候補
> task070@52・task134@320・task202@20・task343@172は、後続のLB実測で全て
> 黒／エラーと確定した。ここにあるcandidate ZIPは提出・マージしないこと。
> 現行の8012.15 authority自体は変更されていない。

This checkpoint contains only candidates whose full audit is already complete. The root champion is unchanged. task161/task175 and the remaining history scan are pending.

| task | cost | half | fresh (2 seeds) | conditional gain | class |
|---:|---:|:---:|---:|---:|---|
| 070 | 66 -> 52 | no | 99.00% / 98.45% | +0.238411 | POLICY95_PRIVATE_ZERO_RISK |
| 134 | 422 -> 320 | no | 96.85% / 96.30% | +0.276684 | POLICY95_PRIVATE_ZERO_RISK |
| 202 | 48 -> 20 | yes | 97.40% / 96.65% | +0.875469 | POLICY95_PRIVATE_ZERO_LINEAGE_NON_GIANT |
| 343 | 173 -> 172 | no | 99.35% / 99.60% | +0.005797 | POLICY95_KNOWN_LB_ZERO_NOT_GUARANTEED |

Confirmed conditional total: **+1.396361** -> **8013.546361**. Only task202 reaches the half-cost target; half-only conditional gain is **+0.875469**.

All four entries are POLICY95 and are not leaderboard-guaranteed. Use `submission_POLICY95_CONFIRMED_CHECKPOINT_NOT_LB_GUARANTEED.zip` for the four-task checkpoint, or `submission_POLICY95_HALF_ONLY_NOT_LB_GUARANTEED.zip` for task202 only.
