# task161 cost186 margin-repaired POLICY90 audit

## Decision

`PASS_NORMAL_POLICY90_RECOMMEND_PROMOTION` against immutable
`submission_base_8009.46.zip`.

- authority: cost190, SHA `5dc274d8515f1ac2a5c58583197984cd60fa2ede69fbe8206992f98940a38fbe`
- candidate: cost186, SHA `57487cce1b40cc7df6097cdf1e82e7bfa53b9bcb6f5be954329ea10d132ced81`
- projected gain: `ln(190/186) = 0.02127739844728488`
- known: 265/266 = 99.6241% in all four ORT configurations
- fresh seed `279161001`: 9925/10000 = 99.25% in every configuration
- fresh seed `279261001`: 9947/10000 = 99.47% in every configuration

The source cost186 model was otherwise eligible but emitted a few positive
values below 0.25 on fresh inputs. The repair changes only initializer `poly`
from its original float32 values to exact float32 `poly * 8`. That tensor is
used exactly once by the final output-producing Einsum, so the entire raw
output is scaled uniformly by positive eight. No node, parameter, memory, or
prediction sign changes.

The repaired graph passes full checking, strict data-propagating shape
inference, canonical static I/O, truthful runtime-shape tracing, and the
no-lookup/cloak/giant/banned/custom/Conv-bias gates. Across four runtime
configurations, runtime errors, nonfinite values, output-shape mismatches,
configuration sign/raw differences, and values in `(0,0.25)` are all zero.
The minimum positive value in the primary fresh audit is
`0.466472327709198`.

Independent review on seeds `280161001` and `280261001` gives respectively
9924/10000 and 9935/10000 in every configuration. It proves that only `poly`
changed, the candidate raw output is exactly source raw times eight on all
review cases, and reports 81,064 candidate executions with all fault counters
zero. See `scripts/golf/agent_review_task161_margin8_280/REPORT.md` and its
`evidence.json`.

This is normal POLICY90, not exact correctness and not a private-zero
exception. Reproduction is `build.py` followed by `audit.py`; primary
machine-readable evidence is `evidence.json`.
