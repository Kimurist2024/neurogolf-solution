# Unsigned comparison identity scan 253

All 400 authority members were scanned for the all-input identities
`uint > 0 == Cast(uint, bool)`, `uint >= 1 == Cast(uint, bool)`, and their
zero-test inverses.  The scan found 81 eligible comparison sites.  Each site
passed full checking and strict inference after rewrite.

No individual or per-task combined candidate is cheaper.  Every removable
threshold scalar remains live in another operator, while inverse zero tests
need an additional Boolean intermediate for `Not(Cast(...))`.  Consequently
the exact rewrite changes no parameter count or increases memory.

Safe adoptees: **0**; projected gain: **+0.0**.  `scan.json` contains all site,
cost, structural, combined-rewrite, and SHA evidence.
