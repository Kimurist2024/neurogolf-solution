# task039 / task111 / task122 fusion audit

## Outcome

**NO SAFE candidate.** All three files are strict-lower by one byte, but all
fail the complete known corpus under the official-scoring execution mode
(`ORT_DISABLE_ALL`). Nothing should be merged.

| task | exact graph change | cost | default ORT | disable-all ORT | decision |
|---:|---|---:|---:|---:|---|
| 039 | remove dead `Equal -> keep_bg_equal` (bool scalar, 1 byte) | 42 -> 41 | 0/264; runtime 264 | 0/264; runtime 264 | reject |
| 111 | remove dead `CenterCropPad -> dummy_13` (int8 scalar, 1 byte) | 89 -> 88 | 265/265; runtime 0 | 0/265; runtime 265 | reject |
| 122 | remove dead `GreaterOrEqual -> d_keep` (bool scalar, 1 byte) | 101 -> 100 | 266/266; runtime 0 | 0/266; runtime 266 | reject |

Counts above are normal accuracy with runtime errors counted as failures. The
required >=90% threshold is therefore missed in the official mode by all three.

## Independent structural and profile checks

For every candidate:

- ONNX full checker: PASS
- strict shape inference with data propagation: PASS
- declared shapes/dtypes versus fresh inference: PASS
- Conv/QLinearConv-family bias UB count: 0
- initializers: bit-identical to authority
- retained nodes: bit-identical; no added nodes
- removed output: unconsumed and not a graph output
- independent faithful-official profile: exactly one memory byte cheaper,
  parameters unchanged

Thus these are graph-semantic dead-end deletions, not arithmetic rewrites. They
are nevertheless unsafe for this model lineage: removing the one-byte output
changes ORT buffer reuse/liveness. The runtime errors report attempted Slice
buffer reuse with incompatible actual shapes, for example:

- task039: `{1,1,1,1} != {1,10,19,19}`
- task111: `{10,1,1,1} != {1,1,30,30}`
- task122: `{1,1,31,30} != {1,30,30,31}`

The 8009.46 authority files run all known examples correctly with runtime0,
nonfinite0, and truthful output shape in both ORT modes. Candidate task039
fails both modes; task111/task122 happen to pass optimized/default ORT but fail
the disable-all mode used by the scorer. Their profile's `correct=false`
independently reproduces the rejection.

## Fresh gate

Fresh seeds 0 and 1 x 5,000 were not run. The complete known runtime gate is a
hard prerequisite and already fails 100% in the official mode; fresh testing
cannot restore runtime0 or >=90% normal accuracy.

## Artifacts and guards

- detailed counters, first runtime errors, exact diffs, profiles, and hashes:
  `audit/result.json`
- reproducible audit: `audit_fusions.py`
- authority copies: `baseline/`
- rejected scan copies only: `candidates/`

Root `submission.zip` and `all_scores.csv` hashes are unchanged. The full
`others/71407` tree digest is also unchanged
(`21d9656e3f4934bcea66c94dca03216c7eba791980bc6ba637a369a60144410a`).

