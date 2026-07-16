# 8005.16 exact no-op/dead-code rescan

**Safe adoptees: 0.**  The full latest payload was run through the repository's
dead-code, identical-initializer, and exact no-op eliminator. Five cheaper
artifacts were emitted, all in known allocator/liveness-risk lineages.

- task039: 42 -> 41; previously complete known-set runtime failure.
- task089: 1349 -> 1180; the member changed in 8005.16, so it was rechecked.
  Both default optimized sessions fail structural loading, and the disable-all
  differential produces runtime shape/buffer errors on all 100 independent
  one-hot probes. It fails runtime0 and truthful-shape gates.
- task111: 89 -> 88; previously complete known-set runtime failure.
- task122: 101 -> 100; previously complete known-set runtime failure.
- task183: 160 -> 89; previously complete known-set runtime failure.

All candidates are rejected. Gain counted is `+0.0`. Evidence:
`manifest_pre_differential.json`; historical confirmation:
`../agent_exact_wave2/REPORT.md` and `../agent_exact_wave2/static_scan.json`.
