# Score-25 similarity scan: cost 51--100

## Result

**NO_SAFE_COST0_OR_COST1_FINALIST**

The immutable authority was `submission_base_8011.05.zip` (LB `8011.05`,
SHA-256 `ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56`).
The scan did not modify `submission.zip`, `all_scores.csv`, or `others/71407`.

## Exhaustive batch completed in this lane

- Scope: all 86 non-score-25 authority tasks whose audited cost is 51--100.
- Generic formulas: 134 per task.  These include every shared zero/one-parameter
  family in `extra15_cost25_scan_294`, the exact authority models for tasks
  067/129/179/241, fixed pooling, unary/binary operators, and the existing
  hand-written input-only Einsum family.
- Task-specific formulas: 85 additional input-only initializer-removal
  sub-equations.
- Total candidate-task evaluations: 11,609.
- Exact-known strict-lower survivors: 0.
- Cost-0/1 finalists: 0.
- Policy-95 finalists: 0 (no candidate passed the known gate far enough to
  justify fresh admission).

Only two formulas scored at all in the 12-case pre-screen, and neither was
close: task380 transpose was 1/12 and task050 identity was 2/12.  Every other
task/formula pair was 0/12 at best.  Full per-task evidence is in
`score25_prescreen.json`; the search is reproducible with `scan_score25.py`.

## Uniform-output lead (the apparent "nine tasks" coincidence)

The nine known uniformly single-color tasks are 021, 049, 056, 100, 103, 129,
291, 339, and 346.  Task129 is the existing cost-1 member.  The same scalar
underflow selector does not transfer safely to the other eight:

- **task056 / task103:** the desired channel is usually absent from the input
  support (41/46 cases for 056, 223/223 for 103).  An input-only or scalar-only
  multilinear formula cannot emit such a channel.
- **task021:** the background color is indeed the unique frequency mode, but
  the output support is a packed, variable `number_of_row_blocks x
  number_of_col_blocks` rectangle.  The task129 power selector supplies the
  color but not this support transform.
- **task049:** the target is the rarest/smallest overlaid rectangle and its
  variable height/width must be packed at the origin.  Task129's positive power
  selector is monotone in frequency and selects the opposite extreme.
- **task100:** the target is the rectangle with larger *area*.  Choosing the
  most frequent nonblack channel matches only 248/266 known cases (93.23%),
  below the 95% policy gate, and it still lacks the fixed top-left 2x2 mask.
- **task291:** the target is the rectangle containing a black hole.  The
  incumbent's 30-element `edge` is what places the scalar decision at canonical
  `[0,0]`; replacing it by free batch axes produces a physical 1x1 tensor, not
  the required canonical 30x30 output.  Padding that result costs at least the
  incumbent's saved elements as runtime memory.  Its 10-element signed color
  vector is also essential to distinguish black holes from solid rectangles.
- **task339:** the color is easy (the only nonblack channel), but output width
  equals the number of colored cells.  A scalar mode selector cannot construct
  the required top-row prefix mask of length 1--9.
- **task346:** the target is the center color of a planted 3x3 local pattern;
  its location-independent detection and fixed top-left cell placement are
  both missing from a task129-style global frequency monomial.  "Choose the
  rarer color" happens to agree in 261/267 known cases (97.75%) but is not the
  true rule and did not yield an executable safe cost-0/1 model.

Thus the number nine is a useful clustering clue, but not evidence that the
eight remaining tasks share task129's clean cost-1 implementation.

## Secondary task229 check

Task229 is the closest genuine task129 analogue: its incumbent one-node Einsum
uses a 17th-power global frequency statistic to retain the unique mode and
recolor other cells gray.  Its symmetric CP factor has rank 4 and costs 40.
Deleting one rank row scores only 4.6%--75.7% over the complete legal count
state space; direct rank-3 optimization reached 70.1%, far below policy95.
Clean cost0/1 is structurally blocked because gray channel 5 is excluded from
generated inputs, so an input-only formula cannot create the required gray
output support.

No candidate from this lane is suitable for promotion.
