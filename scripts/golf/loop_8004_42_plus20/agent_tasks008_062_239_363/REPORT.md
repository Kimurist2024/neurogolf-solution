# tasks008/062/239/363 strict-lower and POLICY90 audit

## Outcome

**Accepted models: 0/4. Projected gain: `+0.000000`.** There is no safe exact
strict-lower candidate, no eligible normal-POLICY90 candidate, and no
all-support pass-through successor for private-zero task363. No submission,
stage, score ledger, or `others/71407` artifact was modified.

The immutable authority is `submission_base_8009.46.zip`, byte-identical to
root `submission.zip`, SHA-256
`4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`.

| task | member SHA-256 | memory | params | cost | known in each of 4 ORT configs | runtime shapes | winner |
|---:|:---|---:|---:|---:|:---:|:---:|:---:|
| 008 | `30abdd1f30f1...` | 331 | 100 | **431** | 266/266 | false | none |
| 062 | `6767dbf75899...` | 370 | 93 | **463** | 267/267 | false | none |
| 239 | `e15519d37cca...` | 328 | 56 | **384** | 267/267 | **true** | none |
| 363 | `5daecf63ed4b...` | 430 | 82 | **512** | 265/265 | false | none |

All four authorities pass full ONNX checking, strict shape inference with
`data_prop=True`, standard-domain, finite-initializer, error/nonfinite, and
Conv-family UB0 checks. Those static checks do not prove runtime shape
truthfulness: tasks008/062/363 deliberately under-declare runtime carriers.

## Generator authority

- task008, `task_05f2a901.py` (`1cc6165c9eba...`): move the perforated red
  rectangle along the separated axis until it touches the cyan 2x2 object,
  with possible flip/transpose.
- task062, `task_2bcee788.py` (`ac2e6be57e97...`): reflect the colored 3x3
  sprite across the red seam and place it on a green 10x10 canvas.
- task239, `task_9af7a82c.py` (`1c16c85b0b53...`): sort distinct color
  frequencies descending and draw top-aligned bars of those heights.
- task363, `task_e5062a87.py` (`81f24acd5c62...`): recover the red exemplar's
  sprite, find every legal translation over black cells, and paint each match.

The four generator files and `common.py` were read directly; the decisions do
not infer rules from the stored examples.

## task008

The cost-431 authority is not eligible as a new safe/POLICY90 lineage: it
declares output `[1,1,1,1]` but returns `[1,10,30,30]`, and several hidden
AffineGrid/ScatterND carriers are likewise under-declared. A reachable explicit
generator case already makes the graph wrong; the independent seed-194008 run
was 1958/2000 (97.9%) in every ORT configuration. This is characterization,
not a candidate pass.

The nominal static-427 history reprices to actual cost 22021; static-430
reprices to 454. The exact one-byte scalar reconstructions for 2, 3, and 5 all
tie at 431 because one removed parameter element is replaced by one charged
int8 activation byte. Rebuilding the sign vector, trimmed coordinate vectors,
shape carrier, theta, or affine-size tensor is equal or larger after required
materialization. No actual strict-lower graph remains, so POLICY90 has no
candidate to evaluate.

## task062

The current cost-463 member is the earlier cost-465 graph after a two-unit
shave. It retains three decisive cloaks: `gn`, `q`, and `qf` are declared
`[1,1,1,1]` but run as `[1,10,30,30]`; a one-example trace contains 63,363
intermediate bytes versus the 370 charged memory profile. Existing fresh
characterization of the semantic lineage is 1990/2000 and 1989/2000, but a
shape-cloaked authority is not a successor candidate.

The comprehensive retained-history scan had 28 alternatives and no numeric
lower graph under the old 465 member. The sole later 463 lead is now the
immutable authority. Post-authority normalization is cost-neutral, and
Add-to-Sum is invalid for uint8. Rank/dtype constraints prevent scalar aliasing;
reconstructing the 3x6 reflection table or shortened coordinate/color vectors
costs more charged activation bytes than it saves parameter elements. There is
no graph below 463 that reaches the POLICY90 pre-gate.

## task239

This is the only clean authority: every runtime tensor matches its truthful
static shape, all 267 known cases pass under disable/default x threads 1/4,
and there are no lookup, domain, UB0, nonfinite, or margin issues.

Its two executable strict-lower reductions are far below POLICY90:

| SHA-256 | cost | reduction | known | fresh characterization | decision |
|:---|---:|:---|---:|---:|:---|
| `d9cc9754b142...` | 374 | remove inactive-feature safety | 24/267 (8.99%) | 75/1000 (7.5%) | reject |
| `a0aecbe36c3d...` | 379 | remove inactive-bar sentinel | 2/267 (0.75%) | 2/1000 (0.2%) | reject |

A third nominal-374 graph cannot execute TopK in ORT. Algebraically, inactive
TopK columns, background within the output rectangle, and zero-hot cells
outside it are three distinct states. The two removed gates encode those
distinctions and cannot be absorbed into a color-only embedding. Factoring the
2x10 embedding and 10x2 decoder still requires a complete materialized table
and does not remove the charged 1x2x12x5 feature field. Exact optimizer passes
produce no lower graph.

## task363

Task363 is in the private-zero/order-sensitive catalog, so normal POLICY90 is
not permitted; a successor must be an all-support pass-through. The current
513-to-512 one-parameter shave is now authority, not a candidate. It has seven
runtime/declaration shape mismatches, including the output, and its two fresh
seeds score 495/500 and 500/500. Thus it is neither truthful nor all-support.

No retained model is below 512; the previous frontier started at 514. All
seven CastLike-to-Cast subsets tie at 512 or rise to 9511. Initializer
reconstruction has a best-case lower bound of `+1`, the biased ten-element
GroupNormalization vector cannot be materialized more cheaply, normalization
loses static shape information, and Add-to-Sum is invalid for int32.

There is also a rule/fixture incompatibility: the truthful pure-rule control is
3000/3000 fresh but 263/265 known at cost 12542. Two stored fixtures conflict
with the generator legality relation, so a deterministic input-only pure-rule
graph cannot satisfy both those fixtures and every legal support point. This
blocks the required private-zero all-support path independently of cost.

## Fresh-gate disposition

No model reaches the intersection of strict-lower cost, checker/strict-data
propagation, truthful runtime shapes, complete pre-gates, allowed structure,
and the applicable exact/POLICY90 rule. Per the stop rule, no new large fresh
run was started. Existing fresh numbers above are retained failure or authority
characterization evidence only.

Machine-readable authority hashes, candidate SHAs, costs, rates, algebraic
dispositions, and final verdicts are in `REPORT.json`; `winner_manifest.json`
is intentionally empty.
