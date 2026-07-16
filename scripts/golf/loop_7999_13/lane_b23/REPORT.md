# Lane B23 — exact singleton-shape Einsum initializer aliases

## Result

No winner was found. The pinned Wave16 archive was not modified.

- Baseline: `submission_7999.13_wave16_candidate_meta.zip`
- Baseline SHA-256:
  `4014cbafea4862f67ebf5ff24be13149b45b333c95bfa680be7216f001a6bb3a`
- Tasks scanned: 400/400
- Initializers inspected: 3,172
- Initializers used exclusively by one or more `Einsum` nodes: 797
- Ordered target/source pairs with equal dtype and element count but different
  rank: 55
- Pairs whose squeezed shapes agree up to an axis permutation: 25
- Byte-exact value aliases after the corresponding permutation: **0**
- Candidates built: **0**
- Winners: **0**

## Exact rewrite screened

`scan_build.py` considers removing a target initializer only when all of its
consumers are `Einsum` nodes. It removes singleton dimensions from target and
source, enumerates dimension-compatible permutations of the remaining axes,
and requires the transposed source to be byte-identical to the target. A
surviving target would be replaced directly with the already-stored source;
only that operand's equation term would change. The construction adds no node,
runtime tensor, or `Einsum` operand.

The subscript builder conservatively rejects ellipses, repeated labels, and
cases where a source lacks singleton axes needed to retain output labels.
Extra source singleton axes receive unused labels and are summed over their
size-one domains. These checks were implemented even though no value-equal
alias reached the builder. `test_scan_build.py` exercises a synthetic
rank-changing/transposed singleton alias; the base and rewritten models pass
40/40 exact comparisons across `ORT_DISABLE_ALL` and default ORT.

Same-rank permutations are intentionally excluded because the parent global
scan already covered that family and found zero. This lane therefore does not
duplicate that search.

## Validation disposition

There is no produced ONNX candidate. Consequently there is no lower-cost model
to send through the checker, truthful static/runtime shape audit, known set,
dual ORT modes, or fresh-5000 gate. Those empirical gates are marked not
applicable, not passed. No candidate and no processing error were accepted.

Evidence:

- `scan_build.py` — pinned, reproducible all-400 scanner/builder
- `test_scan_build.py` — synthetic checker and dual-ORT proof test
- `scan_build_manifest.json` — per-task and aggregate scan counts
- `winner_manifest.json` — empty winner set and validation disposition
