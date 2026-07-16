# task366 shared-ln2 follow-up screen

## Outcome

No additional candidate is admissible beyond the already staged exact
task366 Selu shaves. Projected incremental gain is `+0.0`.

## Trace result

`trace_logs.py` inspected the 21 `Log` outputs divided by the shared `ln2`
initializer on 4,000 generator cases. Four inputs inherit authority runtime
errors. Many of the successfully traced sources reach `-inf`; only five log
sites (`142`, `152`, `236`, `246`, and `459`) stayed finite and nonnegative in
the screen.

Replacing only those five divisions cannot remove `ln2`, because the same
initializer remains live at the other 16 sites. Replacing every division by
Selu is not exact: valid executions reach negative-infinite log values, where
Selu's exponential branch differs from linear division. There is no duplicate
scalar initializer to alias.

No model was staged and the immutable root authority and score ledger were not
modified.
