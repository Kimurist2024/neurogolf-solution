# Staged-candidate optimizer scan 172

All eleven staged SOUND candidates were re-run through 28 conservative ONNX
elimination/fusion passes individually and as a fixed-point set (319 profiles).
Only task192 became strictly cheaper: `eliminate_duplicate_initializer`
reproduced SHA `51a7d654...` and cost1197->1195. Nine optimizer variants
inherited structural failures and were rejected; every other valid profile was
a tie.

Safe adoptees: **1** (the independently audited task192 SHA); incremental
projected gain: **+0.0016722411923621**. Evidence: `scan.py`, `scan.json`, and
`../root_task192_hist_170/REPORT.md`.
