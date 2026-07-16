# agent_task344_deep110 — task344 decoded local-rule deep audit

## Result

**Winner: 0.**  Immutable authority task344 is cost **197**.  Five new
strict-lower models at costs 170--188 satisfy the requested no-S, no lookup,
non-giant, truthful-shape, UB0, nonfinite0 structure, but none passes all known
examples.  Therefore no model reached the independent fresh candidate gate and
no submission, score ledger, `others/`, or root artifact was changed.

## Immutable authority

- archive: `submission_base_8008.14.zip`, SHA-256
  `50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6`;
- task344 SHA-256
  `d0902dc6498525c5f62f12fc02e25fe7914afbae4a583fd77b71f8f05f08019f`;
- official scorer: **0 memory + 197 params = cost 197**;
- known: 266/266 in ORT_DISABLE_ALL and 266/266 in default ORT;
- runtime shape mismatches 0, Conv UB findings 0, standard domain.

The authority is retained as the immutable score threshold, not as a new
structurally eligible result: its only node is a **25-input Einsum** and its
initializers include `S[3,3]`.  All factor matrices are at full structural
rank: H=3, V=4, B=4, S=3, M=4.

## Decoded true rule

All updates read the original input simultaneously:

```text
center == 2 and an orthogonal neighbour is 3  -> 0
center == 3 and an orthogonal neighbour is 2  -> 8
otherwise                                      -> center
```

The readable implementation agrees with all 266 known cases and two new,
independently seeded generator streams:

- seed 344110037: 5000/5000;
- seed 344110091: 5000/5000.

No examples, coordinates, outputs, or fixture lookup are used by the rule.

## Reused historical failures

The previous `agent_sound192_344_93` no-S candidate was re-audited rather than
retrained blindly:

| model | cost | known disable/default | structural reason | decision |
|---|---:|---:|---|---|
| no-S distilled | 188 | 0/266, 0/266 | 24-input giant Einsum | reject |
| shared-V | 181 | 13/266, 13/266 | retains S; 24 inputs | reject |
| spatial-rank-3 | 191 | 266/266, 266/266 | retains S; 24 inputs; fresh inexact | reject |

The rank-3 model's decisive independent evidence is 4972/5000 in each ORT
mode.  Its 28 fresh failures prohibit generator-exact admission even though it
passes the known corpus.

## New non-giant no-S search

`search_compact.py` replaces the authority's fourth-power spatial contraction
with a second-power factored kernel.  This needs eight B operands instead of
sixteen, allowing a maximum of 16 total Einsum inputs.  Five color/spatial
factorizations were trained only from generator output, a fixed independent
generator set, and the known corpus as a numerical guard.

| candidate | cost | max inputs | known disable/default | result |
|---|---:|---:|---:|---|
| shared H + full M | 188 | 16 | 0/266, 0/266 | reject |
| split H/T + diagonal M | 188 | 16 | 0/266, 0/266 | reject |
| split H/T, no M | 184 | 15 | 0/266, 0/266 | reject |
| color-rank-3 + full M | 177 | 16 | 0/266, 0/266 | reject |
| spatial-rank-3 + full color | **170** | 16 | **224/266, 224/266** | reject |

Every emitted model passes:

- full ONNX checker and strict data-propagating shape inference;
- official cost profiling;
- standard domain, no functions, sparse tensors, nested graphs, lookup ops, or
  banned ops;
- no `S` initializer and no node with more than 16 inputs;
- declared/runtime output shape mismatch 0;
- runtime errors 0 and nonfinite output values 0 in both ORT modes;
- Conv UB findings 0 (the new graphs contain no Conv).

The closest cost-170 architecture received an extended search with 5000 new
training cases, 1500 fixed-generator cases, 1800 independent validation cases,
120 epochs, and two restarts.  Its best recorded validation/guard mismatch was
304/67 cells, and the serialized best still failed 42 of 266 known examples.
This is a structural boundary-separation failure, not an ORT or shape failure.

## Structural limit reached

The authority achieves zero intermediate cost by keeping the entire nonlinear
local rule in one giant contraction.  Under the requested restrictions:

- removing S while sharing its H factor loses the independent neighbour/local
  color bases (cost188 reproduces the 0/266 failure);
- preserving independent color bases and full-rank M requires more parameters
  than the 197 threshold before a non-giant spatial implementation is exact;
- reducing the spatial basis to rank3 is the only large parameter release, but
  it cannot separate all boundary positions—consistent with the historical
  4972/5000 result and the new 224/266 result;
- every authority factor is full rank, and its padded B tail cannot be deleted
  while keeping a direct free `[1,10,30,30]` output.

The smallest independently verified no-S/non-giant SOUND control remains the
direct one-node Conv at **cost 910**.  It passes 266/266 in both ORT modes,
truthful shapes, and UB0, but is 713 above the authority and cannot be adopted.

## Fresh and winner gate

Candidate fresh testing requires all of the following first:

1. cost `<197`;
2. known 266/266 in default and ORT_DISABLE_ALL;
3. no S/lookup/cloak/UB/giant contraction;
4. checker, strict inference, runtime0, and nonfinite0.

All five new models pass 1, 3, and 4 but fail 2.  Consequently the required
candidate fresh `2 seeds x 5000 x both ORT modes` was not run; performing it
after a known failure cannot produce a winner.  The two 5000-case runs above
validate the decoded reference rule, not an ONNX candidate.

Machine-readable evidence is in `final_audit.json`, the five
`search_*.json` files, and `winner_manifest.json`.
