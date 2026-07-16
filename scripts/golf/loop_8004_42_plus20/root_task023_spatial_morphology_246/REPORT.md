# task023 spatial morphology census — fail-closed report

## Decision

`winner = null` (`REJECT_KNOWN_GATE`).

The requested two-stage spatial morphology family was implemented and all 48
shape-preserving padding orientations were searched.  It produced a genuine
cost-1621 graph and one screen model above 90% fresh accuracy, but no model
passed the mandatory stored-example gate.  Nothing from this lane is eligible
for root/stage/submission adoption.

- authority task023 cost: `1622`
- clean source: `root_task023_tune80/task023_ranker_coordinate2.onnx`
- clean source SHA-256:
  `763c7002607625de9812bab2ea0cd6db73799a3dc7c8153eaf6c4ee7c7a1d346`
- required known gate: `266/266` in raw/sanitized × disabled/default ORT
- best reproducible known result: `261/266` in all four contexts
- required fresh gate: two independent streams × 10,000, each at least 90%
- winner: none

The full machine-readable record is in `evidence.json`; the per-orientation
training record is in `screen.json`.

## Architecture and measured cost

The single bias-free `[1,1,6,6]` QLinearConv was replaced by:

1. a bias-free `[2,1,4,4]` QLinearConv producing `[1,2,6,6]`; and
2. either a bias-free `[1,2,2,3]` or `[1,2,3,2]` QLinearConv producing the
   original `[1,1,6,6]` score map.

The output assembly, TopK selection, width gate, and box painter are unchanged.
Both saved exploratory graphs measure:

| component | value |
|---|---:|
| memory | 1175 |
| parameters | 446 |
| total cost | **1621** |

The delta from the cost-1541 source is exactly 72 bytes of hidden activation
plus 8 extra weights: `1541 + 72 + 8 = 1621`.  Each QLinearConv has eight
inputs, so neither convolution carries a bias or a short-bias UB condition.

## Complete padding census

To keep both intermediate and final maps at 6x6:

- the 4x4 stage has total padding 1 on each spatial axis;
- the 2x3 stage has total padding `(1,2)`; and
- the 3x2 stage has total padding `(2,1)`.

All splits across top/bottom and left/right were enumerated: 24 for the 2x3
orientation and 24 for the 3x2 orientation, 48 total.  The common screen used
6,000 generated fitting cases and a disjoint 1,500-case stream at seed
`256023001` (base seed `246023001 + 10,000,000`).

| result | layout | known | screen fresh | verdict |
|---|---|---:|---:|---|
| best fresh | `B3x2_p1_0110_p2_2100` | 253/266 | 1376/1500 = **91.733%** | reject known |
| best initial known | `B3x2_p1_1001_p2_2100` | 257/266 | 1321/1500 = 88.067% | reject known |
| integer-refined known | `B3x2_p1_1001_p2_2100` | **261/266** | 1294/1500 = 86.267% | reject both |

The 91.733% result was reproduced exactly in raw/sanitized × disabled/default
ORT with zero runtime errors, nonfinite outputs, or output-shape mismatches.
It is not a qualifying holdout result because the same model misses 13 stored
examples.  The integer-refined model misses five stored examples in every
runtime context.  Known-only multi-restart, teacher-score distillation, and
warm differentiable/integer repair did not close the known gap.

The two requested independent 10,000-case audits were deliberately not run:
the known gate is a prerequisite, so spending those audits on an already
ineligible model could not change the decision.

## Structural and policy audit

Both retained exploratory graphs have the same clean structural result:

- ONNX full checker: pass;
- strict shape inference with data propagation: pass;
- all inferred dimensions static and positive;
- 30 traced node outputs with zero static/runtime shape mismatches;
- canonical `[1,10,30,30]` input and output;
- standard domains, zero functions, zero sparse initializers, zero nested
  graphs, and zero banned operations;
- Conv-family bias checker: zero issues;
- morphology parameters: exactly 44 integer weights;
- lookup/private-correction count: zero.

The learned tensors are two shared spatial kernels.  There is no example bank,
coordinate output table, TfIdfVectorizer, Hardmax, private-zero route, or
per-example correction branch.

## Files

- `search.py`: all-orientation differentiable and integer quantization search;
- `refine.py`: exact known-constrained integer coordinate refinement;
- `finetune.py`: warm-start differentiable repair experiment;
- `finalize.py`: deterministic structural/runtime/cost evidence builder;
- `screen.json`: 48-orientation census;
- `evidence.json`: final all-four-runtime and structural evidence;
- `task023_spatial_morphology.onnx`: rejected best-known artifact (261/266);
- `task023_spatial_fresh_best_rejected.onnx`: rejected best-fresh artifact
  (1376/1500, but 253/266 known).

No Kimi process was used.  Root, stage, authority archives, and submission
archives were not modified.
