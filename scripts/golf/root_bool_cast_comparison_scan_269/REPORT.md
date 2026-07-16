# Boolean cast/comparison scan 269

All 400 immutable 8009.46 members were scanned for a numeric `Cast` or
`CastLike` of a Boolean tensor immediately compared with a scalar constant.
Each surviving site was reduced by its complete two-row truth table to either
`Identity(bool)` or `Not(bool)`.

- comparisons inspected: 905
- Boolean-cast/scalar sites: 7
- exact `Identity`/`Not` identities: 7
- strict-lower profiles: **0**

Five sites belong to task364, one to task366, and one to task369.  The casts
are shared at six sites, so removing the comparison use cannot eliminate the
cast.  At the remaining task364 site, exposing the truthful Boolean tensor
raises the measured runtime activation profile from cost685 to1584.  The
task366 rewrite is cost-neutral (7987), while the task369 rewrite exposes a
similar hidden-shape cost increase (130 to1029).

No candidate reached the cost gate, so runtime/fresh validation and promotion
were correctly skipped.  Root submission, stage, and score ledgers were not
modified.  Full details are in `scan.json`.
