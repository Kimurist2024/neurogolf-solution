# Wave423: cost 51–100 image-guided low-cost transfer

Status: complete, no candidate admitted.

The 51–100-cost authority scope was exhaustively compared against the
cost-1–20 template families (crop/pad, transpose, channel-map, paint and
finite operator reductions).  The corresponding `artifacts/task_viz/task*.png`
images were inspected as a visual cross-check; they show ARC object transforms
whose geometry/color rules are task-specific, not a shared low-cost template.
The sampled task057 image, for example, is a shape extraction/re-encoding rule
with per-example color substitution and cannot be safely transferred by a
static crop/pad or transpose.

All generated variants failed at least one strict gate (known exactness,
strict-lower cost, or fresh exactness).  No ONNX candidate met the requested
90% admission rule with zero runtime/nonfinite/shape/small-positive failures.
Known LB-black tasks 070 and 202 were excluded up front; no files in the root
submission, score ledger, or `others/` were changed.

The prior exhaustive machine scan remains reproducible in
`agent_cost11_100_lowcost_patterns_401/pattern_scan.json` and covered all 148
authority tasks in the 11–100 cost band (22,280 template evaluations), with no
safe candidate after current black exclusions.
