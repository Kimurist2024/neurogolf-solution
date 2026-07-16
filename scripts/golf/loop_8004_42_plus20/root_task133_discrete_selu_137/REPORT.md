# task133 discrete-domain Selu audit 137

## Result

No candidate was admitted. The root submission and staged71407 candidates were
not modified by this lane.

The authority's `scale_m1` is `Round(Sqrt(main_vals))-1`. Under the task133
generator, each per-color count is between0 and16, so the reachable float16
values are exactly a subset of `{-1,0,1,2,3}`. The sole `Mul(scale_m1, .5)` can
therefore be represented by Selu with `gamma=.5` and
`alpha=1/(1-exp(-1))`: the positive branch is `.5*x`, while the only negative
input `x=-1` maps to exactly `-.5`.

The candidate SHA is
`2f42d8fad2d3fadf4efcc4b96809877da881713f8dffd030d08aa1e13dc2fd3f`.
It passes full checker, strict data propagation, and UB0, and profiles
4393->4392. All five reachable binary16 values are bitwise identical under
both ORT modes. Known267 are also raw-identical and correct in four runtime
configurations.

The authority itself has30 runtime/declaration shape contradictions. On two
independent2500-case streams, both authority and candidate share43 and40
runtime failures per mode, respectively. Although successful cases are
raw-identical and shared truth rates are97.04%/97.20%, the inherited runtime
errors violate the campaign's no-error gate. The candidate was rejected and
not staged.

Evidence: `build.json`, `audit.json`, `build_candidate.py`, and
`audit_candidate.py`.
