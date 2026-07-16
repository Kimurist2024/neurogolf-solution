# New low-cost target exact scan

**Safe adoptees: 0.**  Twelve previously unreported 8005.16 members in the
150--500 cost range were scanned for byte-identical initializer aliases,
output-unreachable payload, isolated internal Identity/no-op Cast/no-op Reshape
removal, duplicate deterministic producers, and unused optional outputs.

Tasks scanned: 020/030/161/175/189/193/195/281/302/304/376/384.
All twelve baselines pass the strict structural precheck, but none contains one
of the mechanically exact reduction opportunities.  No candidate was emitted
and gain counted is `+0.0`.

Evidence: `scan_report.json`.
