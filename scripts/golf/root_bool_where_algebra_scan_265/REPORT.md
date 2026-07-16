# Boolean Where algebra scan 265

The all-400 archive contains 1,039 `Where` nodes but no node whose condition,
true branch, and false branch are all Boolean tensors.  Therefore none of the
exact `And`/`Or`/`Not`/`Expand` Boolean selector reductions applies.

Safe adoptees: **0**; projected gain: **+0.0**. The reproducible census is in
`scan.py` and `scan.json`.
