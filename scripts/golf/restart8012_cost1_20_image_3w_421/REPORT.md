# Wave 421 — cost 1–20 image-template transfer to cost 21–100

## Scope

This lane used the rendered ARC panels under `artifacts/task_viz/taskNNN.png`
for the low-cost template set and screened authority tasks with cost 21–100.
The four known LB-black candidates (070@52, 134@320, 202@20, 343@172),
private-zero catalog entries, and already-admitted tasks were excluded.

## Visual classification

Manual inspection of the low-cost panels (including task043, task067, and task135) shows
three dominant families: (1) output-only replication/tiling and color-gather
patterns (task043), (2) fixed-cell extraction/downsampling (task067/task135), and
(3) geometric transpose/flip/expansion families.  These are already covered
by the existing finite template scanner in
`agent_cost11_100_lowcost_patterns_401/scan_patterns.py`.

## Result

No new candidate was admitted.  The prior exhaustive finite-template run
evaluated 22,280 transplanted task graphs over the 148 tasks in the 11–100
range; none was exact on the complete known corpus.  Reusing those templates
for the current 21–100 scope therefore produced zero candidates before the
expensive fresh gate.  No ONNX, submission archive, score ledger, or
`others/` checkpoint was modified by this lane.

The existing POLICY95 rows task202@20 and task070@52 remain excluded because
they are known private-zero/LB-black lineages.  This lane found no safe
POLICY90 (or stronger) strict-lower replacement.

## Safety gates

No candidate reached profiling, so there are no fresh accuracy claims.  Any
future candidate from this scope must pass strict ONNX checking, canonical
I/O, known and fresh accuracy >=90%, and zero runtime errors, nonfinite
outputs, shape mismatches, and small-positive outputs before admission.
