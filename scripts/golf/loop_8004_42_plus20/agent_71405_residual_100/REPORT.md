# 71405 residual exact-SHA reclassification

## Outcome

The updated LB-verified oracle is `submission_base_8008.14.zip`, SHA-256
`50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6`,
MD5 `db4da5cc59186b26572a380725bc2fdf`. It has 400 unique task members.

Per the authority update, the pending expensive fresh run was stopped before
execution. Candidates were reclassified only by exact member SHA. This lane
does not fix or promote any model.

## Exact oracle matches

| task | candidate SHA prefix | oracle result |
|---:|---|---|
| 019 | `e8d7c5ca20fa` | already fixed in 8008.14 |
| 035 | `82b9e298e974` | already fixed in 8008.14 |
| 168 | `642cba5c350b` | already fixed in 8008.14; new LB evidence supersedes the old lineage warning |
| 182 | `625b31492d91` | already fixed; the two 71405 filenames are byte-identical duplicates |
| 191 | `76795962c336` | already fixed in 8008.14 |

These five task changes account for `+0.010541548391599924` of gain already
contained in the 8008.14 oracle. They add zero new residual gain.

## Rejections

- task198 SHA `a18d9d...`: explicit black exclusion; oracle retains
  `4e37cc...`.
- task201 SHA `46f857...`: explicit black exclusion; oracle retains
  `fb28f6...`.
- task208 SHAs `2e2e6f...` and `3d6e01...`: both explicit black exclusions;
  oracle retains `6c9bad...`.
- task251 SHA `ab6bff...`: not present in the verified oracle and also retains
  the QLinearConv bias-length UB; oracle retains `57f557...`.

Machine-readable evidence is in `oracle_manifest.json`. The original
`others/71405` pool, both authority ZIPs, root submission files, CSVs, and
fixed artifacts were not modified.
