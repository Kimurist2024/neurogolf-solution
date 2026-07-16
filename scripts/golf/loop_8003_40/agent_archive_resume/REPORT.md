# Archive resume final report (8003.40 baseline)

## Outcome

- Strict isolated candidate: **task109**
- Candidate SHA-256: `2e7be8671e2e8abe9d3f2f77f0b068f54a70a584ce477affb28fee6372bd25ef`
- Baseline task SHA-256: `afa9dd95632729064fae4f731c78c4bcbcbd5382499c81b467625e9b20ded2f5`
- Truthful official-like cost: `406 -> 405`
- Cost reduction: `1`
- Projected gain: `+0.002466092495193`
- ZIP merge: **not performed**

## Safety gates

- Known: `266/266`, wrong `0`, errors `0`
- Fresh generator, ORT_DISABLE_ALL: `5000/5000`; runtime/output failures `0`
- Fresh generator, default ORT: `5000/5000`; runtime/output failures `0`
- External differential: `45/45` raw-equal executable cases; threshold mismatches `0`; asymmetric errors `0`
- Full ONNX checker: PASS
- Strict shape inference with data propagation: PASS
- Conv-family bias-length UB: `0`
- Functions: `0`; nested graphs: `0`

## Independent semantic proof

After clearing only `graph.value_info`, candidate and baseline deterministic protobufs are byte-for-byte identical. The sole annotation difference is `state_rows_pad`: `[1,1,1,2] -> [1,1,1,1]`. Nodes, attributes, initializers, graph I/O, opsets, functions, and metadata are unchanged, so this candidate introduces no new lookup or task rule.

## Residual archive screening

The static archive had `423` candidates. Annotation-only scanning found task20/task228 variants at the same truthful/static cost, so they were not adopted. task109 is the only cost-reducing annotation-only result. task254 cost42 was rejected: it is a policy-disallowed 33-input giant Einsum and differed from the baseline on 412/500 external threshold cases, despite passing generator fresh5000.
