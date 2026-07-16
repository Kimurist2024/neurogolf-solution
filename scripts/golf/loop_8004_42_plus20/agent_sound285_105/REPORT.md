# agent_sound285_105 — task285 SOUND true-rule reconstruction

## Result

**Safe winner: 0.**  The exact current task285 threshold is **8623**
(`8322 memory + 301 params`).  The smallest complete generator-derived,
fixture-lookup-free, runtime-shape-truthful model found is **14685**
(`13016 memory + 1669 params`), so it is **6062 above** the admission threshold.

No probe candidate exists.  The required fresh `2 seeds x 5000` adoption run
was therefore not started: it is downstream of the strict-lower pre-gate and
cannot make a cost-regressive model admissible.  No root ZIP, `others/`, score
ledger, or submission member was changed.

## Current authority and unsafe-lineage exclusion

Read-only authority:

- archive: `submission_base_8008.14.zip`, SHA-256
  `50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6`;
- task285 member SHA-256
  `366212e29105fde0295030f3ec3bb014bd300f23aa8259ccd79da2eea720b9e2`;
- serialized bytes 19239; measured cost 8623;
- ORT_DISABLE_ALL known: 265/265;
- default ORT: session construction fails because a Concat has inferred first
  dimension 3 but declared first dimension 1.

The current member is retained only as the immutable score threshold.  It is
not treated as SOUND and is not a source for a new candidate.  The full compact
lineage is prohibited because:

- its earlier cost-8717 ancestor failed 7/100 generator cases;
- another 29/30-fresh member caused LB `6844.55 -> 6830.41` (-14.14);
- the lineage contains fixture row masks, approximate row beams/prefix closure,
  and declared/runtime-shape shortcuts.

This satisfies the instruction not to re-admit any prior fresh-1/30 or LB
regression candidate.

## Generator-derived rule

The generator creates one to three Chebyshev-separated sprites.  Each has a
2x2 four-colour hub and a rooted creature of 4--8 cells inside a 5x5 support.
Input shows one full creature quadrant and only the non-black root cells of the
other quadrants.  Output reflects the shown connected support into every
non-black quadrant and recolours each copy with its hub colour.

The complete model uses only this rule:

1. detect every hub using one full vertical occupancy column plus all four
   orthogonal colour inequalities;
2. decode at most three roots, shown angles, and four dynamic hub colours;
3. isolate the root-connected local 5x5 support with six masked 8-neighbour
   bitboard steps (five is insufficient on four reachable shapes);
4. reflect/recolour the support and assemble exact uint32 output row words.

Saved diagonal-first fixtures require 8-neighbour evidence even though the
random generator's default creature builder is cardinally connected.  A raw
5x5 occupancy window is not exact because another separated sprite can still
enter it; the connected flood cannot be removed.

## Minimum SOUND candidate

Lane copy:
`candidates/task285_sound_dedup_copy.onnx`

- SHA-256
  `3e10bc4d23b8692c0c52893ef140e0df45c96ae3c28f13651b19f59325eb7837`;
- cost **14685 = 13016 memory + 1669 params**;
- 203 nodes, 55 initializers, standard ONNX domain only;
- full checker and strict data-propagating shape inference pass;
- declared/runtime shape mismatches: **0 over every intermediate on all 265
  known cases**, in both ORT_DISABLE_ALL and default ORT;
- known correctness: **265/265 in each ORT mode**;
- nonfinite values: **0 over all floating intermediate evaluations** in each
  mode;
- Conv/ConvTranspose UB findings: **0** (neither op is present);
- no banned op, nested graph, giant Einsum, sparse initializer, dynamic shape,
  or nonstandard domain.

The only ScatterElements uses are generic dynamic `(colour,row)` packed-row
assembly.  The largest constants are a mathematical 30x30 successor matrix,
zero scratch buffers, powers of two, and a 64-entry bit-reversal table.  There
are no example outputs, fixture coordinates, fixture masks, or task-instance
lookup rows.

The model is a formal all-input rewrite of the prior cost-14699 SOUND graph:
five groups of byte-identical TensorProto initializers were aliased, saving 14
parameters.  Raw outputs match the source bit-for-bit on 265/265 known cases in
both ORT modes.  The rewrite therefore inherits the source evidence:

- exhaustive hub detector: 23353/23353, including all 23088 reachable
  shape/show/missing configurations and 265 saved cases;
- detector true-generator stream: 10000/10000;
- full model: 1000/1000;
- independent NumPy reference: 3000/3000.

## New exact-rewrite sweep

`sweep_exact_passes.py` applied 15 conservative input-independent standard
optimizer passes individually and in two combinations.  There were only three
unique serialized computations:

- unchanged minimum: 14685;
- serialization-only `eliminate_identity`: 14685;
- `fuse_consecutive_unsqueezes`: 14687.

No standard identity/dead-code/no-op/duplicate-initializer pass reduced the
14685 cost further.

## Concrete lower bounds

These are decomposition-specific bounds, not a universal theorem over every
possible ONNX graph:

- The exact hub detector is measured at **2899**.  Exhaustive enumeration of
  all 1443 reachable creatures x 16 show/missing states proves that all four
  orthogonal colour comparisons are individually necessary.
- The smallest clean packed output tail is measured at **5328**; its last two
  exact uint32 row tensors alone are 1200 bytes each.  Direct shaped ScatterND
  is 1197 cost worse because `[90,2]` int64 indices alone cost 1440 bytes.
- Consequently, the known exact-detector + clean-tail decomposition spends at
  least **8227**, leaving only **396** below 8623 for all root coordinates,
  angle selection, six-step connected component recovery, four hub colours,
  grid validity, and routing.  The complete measured implementation needs
  14685.
- The structurally different shown-colour decomposition has a stronger
  measured pre-detector floor: width/occupancy/colour prefix 1970 plus the
  minimum component-flood/output tail 7374 = **9344**, already above 8623
  before its dynamic selected-colour pack, root detector, coordinates, or any
  parameters.
- The sparse root+angle front end alone is 6033.  Under the current threshold
  it leaves 2590, less than even the clean output tail, before component and
  colour recovery.
- The only one-channel CNN family that profiles below this range is formally
  insufficient: off-centre receptive fields omit a required +/-9 endpoint;
  centred fields compress a varying legal hub colour into one scalar whose
  affine thresholds cannot isolate all three required colours.

Thus known local pruning, tail replacement, shown-colour selection, sparse
root detection, and feasible one-channel CNN routes are closed.  A future
strict-lower SOUND result needs a new primitive/decomposition that fuses
dynamic component recovery, colour routing, and terminal expansion; it cannot
come from another deletion in the audited graph.

## Gate decision

No model simultaneously satisfies:

1. cost `< 8623`;
2. all known cases in default and ORT_DISABLE_ALL;
3. lookup-free rule provenance and truthful runtime shapes;
4. UB0 and nonfinite0; and
5. fresh seed-A 5000/5000 plus seed-B 5000/5000.

The minimum SOUND model satisfies gates 2--4 and has stronger inherited rule
evidence, but fails gate 1 by 6062.  Therefore the lane emits no winner and no
probe manifest entry.

Machine-readable evidence is in `final_audit.json`, `sweep_exact_passes.json`,
and `winner_manifest.json`.
