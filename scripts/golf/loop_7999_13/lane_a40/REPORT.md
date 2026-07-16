# A40 task396 SOUND rebuild audit

## Outcome

- Decision: **NO_ADOPTABLE_CANDIDATE**
- Required threshold: **cost < 1019**
- Score gain: **+0.00**
- Authority task396 SHA-256: `ce0bd7c49e11cbde341756993a71618c5c0bf8e086de6caf56ad93e8588e1d94`
- Authority ZIP SHA-256: `74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534` (unchanged)

The generator rule was independently replayed on all 266 visible cases and
5,000 fresh cases: find the uniquely widest/tallest hollow rectangle, crop it,
and repaint every nonzero crop cell with the other nonzero color.

## Cost/correctness boundary

| model | cost | known | fresh | decision |
|---|---:|---:|---:|---|
| authority | 1019 | 266/266 | 4954/5000 dual | retain current LB-white member |
| rule k2 | 875 | fail | — | reject |
| rule k3 | 939 | fail | — | reject |
| rule k4 | 1003 | fail | — | reject |
| rule k4 occupancy | 1014 | 266/266 | 9906/10000 | reject; heuristic is not SOUND |
| rule k5 | 1067 | fail | — | reject |
| rule k6 | 1131 | fail | — | reject |
| rule k7 | 1195 | 266/266 | 1000/1000 | over threshold |
| corner micro | 1245 | 266/266 | 5000/5000 dual | over threshold; lookup-like ops |
| full spec runs | 16457 | 266/266 | 1000/1000 | over threshold; lookup ops |

The previous cheapest generator-SOUND control remains cost 1245.  Its exact
corner stage must retain all possible top-left rows before width selection;
reducing to 2–6 high-count rows is not generator-entailed.

## New low-cost decomposition check

I tested five one-linear-row-score replacements over 20,000 fresh
generator cases.  Even retaining the top four rows, failures were:

- Laplacian score: 169
- absolute difference: 1625
- peak score: 282
- minimum-edge score: 1619
- plain count: 338

Projection overlaps from the other one or two rectangles can outrank either
border of the largest rectangle.  Thus a cheap linear score cannot replace the
nonlinear same-color corner/run test soundly.

No candidate survived the cost + SOUND gates, so no new known-dual/fresh-dual
5000/external-500 promotion run was warranted.  The shared ZIP was not changed.
