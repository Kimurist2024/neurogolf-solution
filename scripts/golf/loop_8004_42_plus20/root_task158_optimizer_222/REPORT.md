# task158 cost-7498 optimizer rescan

The newly promoted task158 SHA
`e7101699bfc022fa794e15d7f374a8febe3e2680b8388c67b9a81cdc9962ced0`
was re-profiled under the same 21 fusion/rewriter sets and four fixed-point
cleanup sets used by the stage-wide lane 216.

- baseline: memory 6638 + params 860 = cost 7498;
- profiles: 25;
- byte-changed optimizer results: 25;
- strict-lower profiles: **0**.

No candidate was emitted or promoted.  `scan.py` and `scan.json` contain the
reproducible per-pass results.  Root submission and score ledgers were not
modified.
