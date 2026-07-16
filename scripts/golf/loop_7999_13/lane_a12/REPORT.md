# A12 strict archive audit

## Outcome

- Exact baseline: `submission_base_7999.13.zip`
- Baseline SHA-256: `a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1`
- Tasks: 198, 200, 201, 219, 302, 343
- Retained candidates screened: 23
- Safe winners: 0
- Verified gain: `+0.000000`

No root ZIP, CSV, score pointer, or ledger was written by this lane.

## Results

| Task | Exact cost | Best nominal cost | Strict result |
|---:|---:|---:|---|
| 198 | 661 | 554 | All eight candidates retain a 24-26 input giant Einsum; structure reject. |
| 200 | 346 | 344 | Cost 344 fails known data. The two cost-345 known passes have a one-element bias for a two-channel Conv; UB reject. |
| 201 | 811 | 543 | All three known passes fail fresh data. |
| 219 | 1479 | 1081 | r04 fails known data; all four other known passes fail fresh data. |
| 302 | 160 | 159 | The nominal saving is a pure `value_info` dimension cloak; reject. |
| 343 | 173 | 172 | Both known passes fail fresh data. |

## Required special audits

### task200 bias and UB

The cost-345 models have Conv weight/output-channel count 2 but supply
`conv_bias` with only one element. The strengthened bias checker reports
`('Conv', 'conv_bias', 1, 2)`. Depending on an out-of-bounds or uninitialized
read is not a valid model optimization. The independent earlier B6 audit
reached the same result. The structurally valid cost-344 sharing model fails
the complete known corpus.

### task302 metadata cloak

The cost-159 r02 candidate has exactly the same nodes, initializers, graph I/O,
and opset as the exact cost-160 member. Its only difference is the `up_1`
`value_info` shape:

- exact: `[1, 1, 1, 2]`
- candidate: `[1, 1, 1, 1]`

Clearing `value_info` and non-executable metadata makes both models byte-equal.
Therefore the apparent one-unit saving is a shape/value-info accounting cloak,
not an executable optimization. r01 is executable but profiles at cost 52355.

## Fresh rejection evidence

Fresh screens use generated gold examples and identical cases under
`ORT_DISABLE_ALL` and default ORT. A single mismatch is sufficient to reject;
only a perfect preliminary screen would proceed to 5000/5000 adoption testing.

| Candidate | Disabled | Default | Count |
|---|---:|---:|---:|
| task201 r01 | 0/500 | 0/500 | 500 |
| task201 r02 | 0/500 | 0/500 | 500 |
| task201 r03 | 4290/5000 | 4290/5000 | 5000 |
| task219 r01 | 32/500 | 32/500 | 500 |
| task219 r02 | 9/500 | 9/500 | 500 |
| task219 r03 | 7/500 | 7/500 | 500 |
| task219 r05 | 417/500 | 417/500 | 500 |
| task343 r01 | 4975/5000 | 4975/5000 | 5000 |
| task343 r02 | 497/500 | 497/500 | 500 |

Every generation run recorded zero generation errors. Complete model profiles,
structure verdicts, executable diffs, and individual fresh JSON outputs are
stored in this lane. No candidate reached the dual-perfect adoption gate.
