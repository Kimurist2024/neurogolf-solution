# task007 cost68 normal-POLICY90 audit

## Decision

`ACCEPT_POLICY90` against immutable `submission_base_8009.46.zip`.

- authority: cost70, SHA `fc02d641241760fe6fa7e7ef1be2ba9aa492e7cfe42d94778ea06016573ce0b3`
- candidate: cost68, SHA `fa22f345634e3f059b0b2d334e6b9d85d60973d5cc2a6c92003b8f7cfc60486a`
- projected gain: `ln(70/68) = 0.028987536873252187`
- known: 260/266 = 97.7444% in each of four ORT configurations
- fresh seed `277007001`: 9775/10000 = 97.75% in each configuration
- fresh seed `277107001`: 9726/10000 = 97.26% in each configuration

The candidate is one output-only, ten-input Einsum with two finite small
initializers. It passes full checking and strict shape inference, has canonical
static I/O, no intermediate memory, lookup, fixture table, shape cloak, giant
contraction, banned/custom op, nested graph, or Conv-family bias finding.

Across all four runtime configurations and all 80,000 fresh candidate runs,
runtime errors, nonfinite values, output-shape mismatches, configuration sign
differences, and values in `(0,0.25)` are all zero. The six known misses and
fresh misses are disclosed; this is normal POLICY90, not exact correctness and
not a private-zero exception.

Independent evidence with disjoint seeds is in
`scripts/golf/agent_review_task007_policy90_278/REPORT.md`: 260/266 known,
9745/10000 and 9752/10000 fresh, with the same zero-fault structural and
runtime result.

Machine-readable primary evidence is `evidence.json`; reproduction is
`audit.py`. The audit itself did not modify root or `others/71407`.
