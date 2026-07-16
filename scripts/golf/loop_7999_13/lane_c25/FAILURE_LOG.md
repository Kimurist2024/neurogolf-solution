# C25 rejected lineages

## task131 archive compression

- r01/r02/r03 are known-set-correct but depend on false tensor declarations;
  r01/r02 additionally contain lookup-style `TfIdfVectorizer` nodes.
- r01 is generator-unsound: existing independent dual-ORT fresh5000 evidence
  records 782 correct and 4,218 runtime/output failures in each mode.
- r04/r05 cannot create an ORT session because their `TopK(11)` type signature
  has no implementation in the pinned runtime.
- Therefore none may be merged even though r01/r02/r03 appear cheaper.

## task251 archive compression

- All four retained archive candidates use large `CenterCropPad` shape cloaks.
- Each has 58–64 declared/runtime tensor-shape mismatches.
- Each fails default-optimization ORT session creation.
- The apparent cost-582 and cost-709 candidates are therefore unsafe.

## truthful rebuild controls

- task131 conventional truthful control: cost 24,927 versus baseline 746.
- task251 conventional truthful control: cost 24,708 versus baseline 755.
- task251 exact QLinearConv rebuild: cost 1,869 and 30 shape mismatches.

No rejected branch was copied into the root ZIP, score CSV/JSON, or handcrafted
artifact tree.
