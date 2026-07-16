# task192 true-rule retraining — fail-closed result

## Outcome

No task192 replacement is safe to adopt. The winner set is empty and projected
gain is `+0.0`.

The immutable comparison base is `submission_base_8005.17.zip`, SHA-256
`c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04`.
Its task192 member is the retained LB-white SHA
`e7f9a11b93b611acfa4bba39e90e1ddf24223d50add4277fe9716f21f6ede10c`,
cost 1609. No ZIP, score ledger, or protected file was modified.

## Best retrained strict-lower model

`candidates/task192_large2k_k4l_k4l_p0p0.onnx` is a new SHA
`57920f7f6a8f3f015d742f4e60e0a7105a781d3e412fd29edbf68f97bd44d551`.
It was trained from the decoded generator rule on four seeds and 2,000 fresh
cases. It uses no output table: the graph selects the most frequent nonzero
color from the input, then uses one 3x4 Conv whose coefficients distinguish
background, selected color, and other nonzero colors.

| gate | result |
|---|---:|
| official-like cost | 1309 = 88 memory + 1221 params |
| local known, disabled/default ORT | 265/265 in each mode |
| official known | pass |
| fresh seed 79192931 | 461/500 = 92.2% in each ORT mode |
| fresh seed 79192932 | 462/500 = 92.4% in each ORT mode |
| runtime errors | 0 |
| known margin | stable, minimum positive 158.480224609375 |

It therefore clears the ordinary user-authorized 90% numerical threshold and
is substantially better than the previous cost-1309 h7904 model, whose minimum
on the same style of two-seed audit was 87.4%.

It is nevertheless **not adopted**. The two fresh audits contain 39 and 38
wrong grids. This is a fitted one-layer approximation of the horizontal-AND-
vertical predicate, not an exact rule compiler.

## Why the 90% pass is insufficient here

`docs/golf/private_zero_tasks.md` explicitly catalogs task192 and records the
h7904 cost-1309 replacement as leaderboard-black. Under the user's current
condition, a private-zero task may pass only when its true rule or complete
support behavior is guaranteed. A new SHA and a 92% fresh rate do not provide
that guarantee. Accepting this model would repeat the known task192 failure
class, so the lane fails closed.

## k3 / k4 / k5 comparison

- k3 has nominal cost 1009, but once channel 0 must be positive on every
  background cell, the exact background-complement separator is infeasible.
  The independent reconstruction in `lane_b31` reached the same obstruction.
- k4 is strictly cheaper (1309) and can exceed 90% after the 2,000-case
  multi-seed retraining, but remains non-exact because one linear threshold
  cannot represent the saturated axis-presence AND on all supported patches.
- k5 is the incumbent family. Its ordinary histogram + dynamic bias + dense
  Conv already costs 1609, so it cannot be a strict-lower replacement without
  dropping context. The alternative Gather-bias retrain costs 1622 and is also
  less accurate.

The decoded predicate is:

1. choose the most frequent nonzero color `A`, with the lower color winning a
   tie;
2. emit `A` at a nonzero center exactly when `A` occurs in both the horizontal
   radius-one and vertical radius-one windows;
3. emit zero otherwise.

That conjunction is nonlinear after each axis's presence is saturated. A
single terminal Conv is only linear. Adding truthful spatial intermediates to
implement both presences and the AND raises memory beyond the cost-1609
incumbent in the tested exact families.

## Exact control

The existing generator-SOUND bitset implementation
`scripts/golf/scratch_codex/task192/pad_axes_probe.onnx`, SHA
`16f59d172be152d14e087e54085d6ef2cb6ee188528e70e9549c1a3fac391193`,
passes known 265/265 and retained independent fresh 5000/5000. Its measured
cost is 3325 (3208 memory + 117 params), so replacing the cost-1609 white
fallback would lose score. It is proof that exact behavior is available, but
not at a score-improving cost.

## Structural evidence

The evidence-only cost-1309 candidate passes full ONNX checker, strict
data-propagating shape inference, static positive shapes, and a runtime shape
trace with zero declared/actual mismatches. It has only standard-domain
`Einsum`, `ArgMax`, `ScatterElements`, and `Conv`; no banned op, nested graph,
function, sparse initializer, lookup, or giant contraction. The inferred
dynamic Conv bias is `[10]` for ten output channels, giving Conv UB count zero.

Evidence files:

- `final_audit.json`: complete fixed-SHA structural, cost, dual-known, and
  two-seed fresh audit;
- `large2k_audit_2x500.json`: independent disabled/default ORT fresh results;
- `train_large2k_k4l_k4l.json` and `train_large2k_k4t_k4t.json`: four-seed 2,000-case
  training comparisons;
- `existing_candidate_audit_2x500.json`: incumbent and historical-candidate
  controls;
- `result.json`: machine-readable disposition;
- `winner_manifest.json`: empty safe winner set.
