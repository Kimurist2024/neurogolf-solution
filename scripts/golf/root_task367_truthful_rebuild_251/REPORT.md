# task367 truthful rebuild — early-stop reference

## Outcome

Duplicate exploration was stopped before candidate generation.  The current
authority and complete historical frontier were already audited in:

- `scripts/golf/loop_8004_42_plus20/agent_high367_382_270_154/REPORT.md`
- `scripts/golf/loop_8004_42_plus20/agent_high367_382_270_154/history_coverage.json`
- `scripts/golf/loop_8004_42_plus20/agent_sound319_367_107/REPORT.md`

Pinned conclusion:

| item | result |
|---|---:|
| authority SHA | `b2b73f9637bd3b45a63256c94e019951625a3ec0d069e3b606394f57923ee8aa` |
| authority cost | 2179 |
| authority runtime-shape mismatches | 65 |
| authority default ORT | session construction failure at `CenterCropPad` |
| best truthful exact control SHA | `7673a580bc645f491eb85b110b142d3c6ed5dcac91df0b676c9556c6b156bdbf` |
| best truthful exact control cost | 3913 |
| truthful control gates | full/strict/truthful/UB0, known 266/266 x4, fresh 20000/20000 |

The exhaustive history covered 901 source references and 69 unique task367
SHAs.  Its minimum profiled historical cost was 2247, already above authority;
no current-authority rescreen survivor remained.  Therefore no truthful
strict-lower candidate exists in the audited frontier, and new fresh runs would
not change the decision.

Accepted candidates: **0**. Projected gain: **+0.0**.

