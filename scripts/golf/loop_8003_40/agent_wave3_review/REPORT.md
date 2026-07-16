# Wave3 independent review

## Verdict

**APPROVE**

Wave3 SHA-256 is
`338b6c968bb345780a570ec849f17b2fc0c1233c5bd0a000c67a035aeafb0cd7`.
Every independently recomputed gate passed.

## Archive identity

- Wave1 SHA-256:
  `c829af1a2928fbbcdae3d13a38a8f38777e4c38c6a08c2c5ae51a3c4f0bd2c49`
- Wave1 → Wave3 changed member: **task109 only**
- Other 399 uncompressed member payloads: byte-identical
- Other 399 compressed member streams: byte-identical
- Member order, invariant `ZipInfo` metadata, per-member comments, and archive
  comment: preserved
- Members/tasks: 400/400, unique and complete task001–task400
- Largest model: task391, 1,506,160 bytes; limit 1,509,949 bytes
- Over-limit models: 0
- `ZipFile.testzip()`: PASS
- Conv/ConvTranspose/QLinearConv bias-length UB across all 400 models: 0
- Model parse failures during the bias audit: 0

## task109 payload review

- Wave1 member SHA-256:
  `afa9dd95632729064fae4f731c78c4bcbcbd5382499c81b467625e9b20ded2f5`
- Wave3 member and standalone candidate SHA-256:
  `2e7be8671e2e8abe9d3f2f77f0b068f54a70a584ce477affb28fee6372bd25ef`
- Both serialized payloads are 1,826 bytes.
- After clearing `graph.value_info`, deterministic protobufs are byte-identical.
  Nodes, attributes, initializers, graph I/O, opsets, functions, and model
  metadata therefore have no difference.
- Sole annotation difference:
  `state_rows_pad` int8 shape `[1,1,1,2] → [1,1,1,1]`.
- Full checker: PASS
- Strict shape inference with data propagation: PASS
- Conv-family bias UB: 0

## Independent execution evidence

- Known: 266/266, wrong 0, runtime errors 0
- Fresh, new seed 109803341, ORT_DISABLE_ALL: 5000/5000, errors 0
- Fresh, same cases, default ORT: 5000/5000, errors 0
- Fresh generation errors: 0
- External differential, new seed 109803342: 31/31 executable cases
  raw- and threshold-identical; mismatches 0; asymmetric errors 0. The remaining
  469 cases failed symmetrically in both Wave1 and Wave3 members.
- Truthful official-like cost: 406 → 405

## Projection

The Wave1 gain was recomputed from its nine individual cost pairs rather than
using its stored summary:

`8003.40 + 0.1355960288516288 + ln(406/405)`

`= 8003.538062121347`

This exactly matches the required Wave3 projection.

No ZIP, builder, protected root file, or `LOOP_STATUS.md` was modified by this
review.
