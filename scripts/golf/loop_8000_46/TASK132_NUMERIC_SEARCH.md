# task132 numerical gauge search

The cost-312 gauge-reuse family remains rejected against the exact 8000.46
member.  The original candidate differs at one threshold case in the pinned
500-case arbitrary-grid corpus.  This pass tested three additional families:

- 30 power/scale-balancing variants;
- 48 condition-number-ranked exact gauges;
- 64 dyadic/simple exact gauges and 48 gauges nearest the original transform.

All models retained the exact repeated-index identity used to remove the
four-element `A` initializer and passed ONNX construction checks.  Candidates
were first screened on the same seed-80004601 corpus; survivors of 50 cases
were rerun on all 500 cases.  Zero candidates achieved 500/500 threshold
equivalence with zero asymmetric runtime failures.  No task132 model from this
search is included in the safe aggregate.

Evidence is in `lane_task132_scale/screen500.json`,
`lane_task132_gauge_search/screen500.json`,
`lane_task132_gauge_simple/screen50.json`,
`lane_task132_gauge_simple/screen500.json`,
`lane_task132_gauge_near/screen50.json`, and
`lane_task132_gauge_near/screen500.json`.
