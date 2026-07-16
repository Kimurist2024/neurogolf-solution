# 8009.46 current memory census for the +50 campaign

Authority: `submission_base_8009.46.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.

All 400 members were reprofiled from the current immutable archive with the
project scorer under ORT optimizations disabled. Because a zero input can
understate runtime shapes for a few data-dependent graphs, canonical task costs
come from the current `all_scores.csv`; parameter counts come directly from the
current members, and memory is canonical cost minus parameters. The aggregate
current cost is 192,234: intermediate memory 155,413 plus 36,821 initializer
elements.

This census identifies **94 tasks** with canonical memory at least 300 and a
theoretical score gain of at least 0.15 if their memory were halved while
parameters stayed fixed. Their summed half-memory upper opportunity is
**+53.7045885244**. This is a prioritization bound, not a promised gain: many
graphs cannot safely halve memory without changing their rule or adding other
intermediates.

The leading current targets by half-memory opportunity include task364, 338,
205, 037, 145, 009, 101, 076, 088, 089, 077, 361, 279, 285, 233, 216, 366,
173, 036, 286, 018, and 133. High-cost/current-SOUND work is prioritized over
blind dtype changes; all replacements still require strict-lower official
profiling, known correctness, runtime safety, and exact/true-rule evidence.

Artifacts:

- `costs.json`: raw zero-input profiles used to measure parameter counts.
- `canonical_costs.json`: all400 canonical cost/memory/parameter rows.
- `mem_targets_8009_46.json`: ranked94-task active target list.

No submission, score ledger, baseline archive, or `others/` candidate was
modified by the census.
