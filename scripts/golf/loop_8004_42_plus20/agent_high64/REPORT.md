# High64 exhaustive history audit

## Decision

**SAFE WINNERS: 0** for `task103,372,073,130,016,017,061,197`.

The immutable baseline is `submission_base_8005.16.zip`, SHA-256
`73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00`.
No submission ZIP or protected root score/submission file was changed.

## Coverage

The retained archive shortlist was checked first.  The lane then scanned every
loose `*.onnx` under the repository, selected explicit task-number paths, and
SHA-256-deduplicated per task.  This produced 137 distinct non-baseline models
from 4,622 path observations.  Thirteen models had a positive official-like
cost strictly below the immutable baseline.

| task | base cost | observed paths | unique non-base | strict lower | safe pre-fresh |
|---:|---:|---:|---:|---:|---:|
| 103 | 15 | 577 | 16 | 3 | 0 |
| 372 | 13 | 588 | 23 | 4 | 0 |
| 073 | 12 | 584 | 18 | 6 | 0 |
| 130 | 11 | 576 | 18 | 0 | 0 |
| 016 | 10 | 565 | 2 | 0 | 0 |
| 017 | 10 | 578 | 22 | 0 | 0 |
| 061 | 10 | 572 | 12 | 0 | 0 |
| 197 | 10 | 582 | 26 | 0 | 0 |

## Strict-lower rejections

- `task103`: the three cost-1/7 leads scored only 19/223, 0/223, or raised a
  runtime error on all 223 known cases.  None reached known100 dual.
- `task372`: four distinct cost-12 leads were found versus cost 13.  Every one
  contains a non-finite initializer and a shortened `ConvTranspose` bias
  (`9` entries for `10` output channels).  These are explicit immediate
  fail-closed rejections.  The three archived variants happened to score
  266/266 in both known runtime modes, but that does not override the
  non-finite/bias safety gate and is not a private-zero guarantee.
- `task073`: one cost-6 lead contains a non-finite initializer.  The other five
  cost-7 through cost-11 truncated-parameter variants score 0/15 known and use
  runtime/declared shape mismatches.
- `task130,016,017,061,197`: no strict numeric lower loose or retained lead.

No model passed strict-lower + known100-dual + truthful structural gates.
Consequently all-known trace repricing, two-seed fresh validation, private-zero
true-rule proof, and ZIP integration were correctly not entered.  Projected
accepted gain is **+0.0**.

Primary evidence: `history_lead_audit.json` and `all_history_audit.json`.
