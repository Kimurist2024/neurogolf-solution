# cost>=5000 harvest (everythingbundle + others) — 2026-06-23

Scanned 1,783 candidate ONNX (102 unique) over the 24 cost>=5000 tasks against
_BEST_7530.01.zip via scan_sources_seq.py.

6 LOCAL winners (correct + strictly cheaper) — ALL fresh-gate REJECTED (k=30):

| task | base | cand | fresh_fails/30 | verdict |
|------|------|------|----------------|---------|
| 023  | 6920 | 1520 | 4 | REJECT (overfit) |
| 118  | 9144 | 7055 | 1 | REJECT |
| 018  | 7805 | 7389 | 2 | REJECT |
| 173  | 12272| 11492| 1 | REJECT |
| 285  | 10910| 10908| 1 | REJECT |
| 286  | 13661| 13636| 2 | REJECT |

Conclusion: every cheaper bundle net is overfit to visible gold (fails the fresh
generator audit) -> would score private 0. NONE adopted. Harvest yielded nothing
safe for cost>=5000. Next lever: spec-compiled from-scratch rebuilds (fresh-clean
by construction).
