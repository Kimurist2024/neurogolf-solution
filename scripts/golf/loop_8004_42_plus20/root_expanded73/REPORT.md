# Expanded private-lineage guarantee pre-screen 73

Three additional lower-cost files were reconsidered after the user allowed a
private-zero lineage only when passage can be guaranteed.  None qualifies.

- task048 candidates `c71df5f5...` and `cdde85d8...` each reduce actual cost
  379 -> 378, but both inherit the same wrong predictions: 4503/5000 and
  4554/5000 on two independent generator seeds in both ORT modes.  The task
  generator ranges over 5--8 by 5--8 grids, arbitrary cyan subsets, and pairs
  of non-overlapping 2x2 boxes; its support is not a tractable finite set that
  has been exhausted.  Existing counterexamples disprove a guarantee.
- task365 candidate `887d5695...` reduces actual cost 1369 -> 1337 and is
  known266/266 in both modes, but independently scores 4596/5000 and
  4563/5000.  The generator permits two or three 3--6 by 3--6 boxes with
  combinatorial per-pixel colors and red placements.  Existing legal
  counterexamples again disprove a guarantee.

Safe adoptees: **0**.  Gain counted: **+0.0**.  No ZIP or protected root file
was modified.
