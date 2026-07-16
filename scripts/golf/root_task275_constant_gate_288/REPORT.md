# task275 constant-gate attempt

Decision: **REJECT**.

The structural rewrite reduced the measured profile from cost428 to cost414,
but its proof premise was false.  The canonical scoring tensor does not put a
one in the color-0 channel for background cells: background is encoded as all
zero channels.  Therefore `ReduceSum(input)` is the number of nonzero-colored
cells, not the constant 900 (the first stored example sums to 18).

Consequently the proposed constant `gate=[-875,7]` is not equivalent to the
authority's dynamic `Conv(total,GW,GB)`.  The four-configuration audit scored
the candidate 0/266 known and 0/500 on each of two fresh streams, with raw
equality 0 throughout.  Runtime, nonfinite and output-shape errors were zero,
but semantic failure is complete.

The rejected model remains isolated under `candidates/` as evidence.  It was
not copied to `others/71407` and does not contribute any projected gain.
