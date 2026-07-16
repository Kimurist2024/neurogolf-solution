# High-cost safe rebuild report (tasks 150-400)

## Outcome

- Ranked 251 baseline tasks using truthful runtime cost.
- Selected the top three sound Type A/B rebuilds after excluding private-zero, unsound, contaminated, other-lane, lookup, shape-cloak, Type C/D, and giant-Einsum tasks.
- Input-only rule references pass **5000/5000 fresh** for task156, task237, and task345.
- Safe lower-cost ONNX candidates: **0**.
- ZIP merge: **not performed**.
- Final verdict: **NO_CANDIDATE**.

| Task | Type | Baseline cost | Clean-floor conclusion |
|---:|---|---:|---|
| 156 | A + two-component extent comparison | 556 (330 memory + 226 params) | Conventional decode/masks exceed the incumbent; 30x30 label/condition alone is >=900 bytes. |
| 237 | B, bounded right/down propagation | 529 (413 + 116) | Honest 9x9 decode + one propagation grid is >=648 bytes before params. |
| 345 | B, nine-step obstacle path | 389 (248 + 141) | Direct boolean unroll is >=900 bytes; incumbent scalar bitsets are already below the clean floor. |

## Concrete task237 probes

- `remove_min`, cost 520: **REJECT**, fresh 3/20. Empty rows require the sentinel clamp.
- `shift_shrink`, cost 528: **REJECT**, fresh 0/20. The unshifted width is also required by `Min`.
- combined, cost 519: **REJECT**, fresh 0/100.

All three files pass full checker, strict shape inference/data propagation, and Conv-family bias-length checks, but fail correctness and are not candidates. Nine serious design/build attempts are recorded in `FINAL_REPORT.json`; further work stops at the structural floor.
