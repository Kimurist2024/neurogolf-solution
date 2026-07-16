# agent_target_mid12 — strict SOUND audit

## Outcome

Accepted candidates: **0**. Projected gain: **+0.0**. No ONNX candidate was
saved because none was simultaneously (a) a true-rule implementation without
lookup/cloak/giant-Einsum dependence, (b) truthful/static and dual-ORT safe, and
(c) strictly cheaper than the exact member of `submission_base_8004.50.zip`.

The baseline ZIP SHA-256 is
`63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac`.
All baseline costs below are the measured costs of that fixed ZIP, not the stale
brief/CSV values. In particular, task330 is **897**, not the stale 896 row.

| task | baseline cost | baseline terminal issue | lowest audited true-rule control | control cost | decision |
|---:|---:|---|---|---:|---|
| 330 | 897 | shape cloak, default-ORT session failure, legal generator counterexamples | `task330_truthful_component_rect` | 5,525 | reject (+4,628 cost) |
| 280 | 828 | 22 shape mismatches, default-ORT session failure, 24-input Einsum, prior fresh 99/100 | truthful beam renderer | 2,161 | reject (+1,333 cost) |
| 364 | 685 | `input_fake` guaranteed runtime shape contradicts declared scalar shape | truthful topology/flood model | 46,741 | reject (+46,056 cost) |
| 310 | 566 | TfIdf lookup + 23-input Einsum; documented fresh/private divergence | exact selector control | 633 | reject (+67 cost and prohibited ops) |

Therefore every `candidate_path`, `candidate_sha256`, and `new_cost` is null in
`result.json`; old/new gain is 0 for all four tasks. No ZIP integration was done.

## Generator truth

The obfuscated Sakana functions were run independently over every stored pair.
They reproduce **266/266** pairs for tasks 330/364/310 and **267/267** for
task280 (all exceed the requested 265-case floor):

- task330: each separated four-connected gray component is red iff its size is
  exactly six; all other gray components are blue.
- task280: each red edge-dot emits a red centerline plus the green thickness
  band outward from its rectangle, with defect direction, flip, and transpose.
- task364: classify each separated green glyph topologically: H/junction -> 2,
  U/two-corner -> 6, L/one-corner -> 1.
- task310: find the unique complete square perimeter of side 5..8 in the
  periodic wire grid and return exactly its crop.

These are component/global-geometry rules. The very low baseline costs are not
sound lower bounds for ordinary truthful graphs: three exploit opaque runtime
shapes, while task310 uses the prohibited lookup/giant-contraction family.

## Per-task evidence

### task330

Baseline `submission_base_8004.50.zip::task330.onnx`, SHA-256
`06dcb5216cb441f6d760a28fa4a5b4affa678c9001504dbd8f4f80f3bbd2d5af`,
cost 897 (memory 730, params 167), uses GroupNormalization, 31
CenterCropPads, three QLinearConvs, and Mod. Strict inference/checker pass only
because the opaque supplied declarations are accepted; they are not truthful:
`g` is declared `[1,1,1,1]` although GroupNormalization preserves the fixed
`[1,10,30,30]` input, and output is declared `[1,10,22,22]` while runtime is
`[1,10,30,30]`.

Recheck: disable-all 266/266, runtime errors 0, margin min 1.0; default ORT
cannot create a session. Two independently constructed generator-legal
close-component cases fail the baseline and pass the exact control. Thus this
is not merely a metadata objection: the local detector is semantically unsafe.

The no-cloak/no-Einsum component-rectangle control is
`scripts/golf/loop_7999_13/lane_a26/rule_references/task330_truthful_component_rect.onnx`,
SHA-256 `6268e9acdeb8a79af2c1dc2485bc843a54d0b599ec41be1de74faa75ac5c610d`,
cost 5,525. It is 266/266 in both ORT modes, errors 0, truthful static output,
strict/data-prop pass, and Conv-family bias UB=0. It is rejected before the
two-seed fresh gate because it already loses 4,628 cost (score delta -1.8180).

### task280

Baseline SHA-256
`19d38b7bff083fd7da14262714afa75345594b91e39d978ada0d25a912971793`,
cost 828 (memory 626, params 202), has 15 Einsums including two 24-input,
eight 21-input, and four 17-input contractions, plus 20 CenterCropPads. It is
267/267 with disable-all, errors 0 and margin min 1.0, but default ORT session
creation fails. The audited lineage has 22 declared/runtime shape mismatches;
a prior fresh100 audit was 99/100, so it is below the mandatory 100% guarantee.

The truthful no-cloak control is
`scripts/golf/loop_7999_13/lane_b17/candidate_task280_truthful.onnx`, SHA-256
`7922cea4b2789ed175357f6c5e3855b4b19521ea90bd6dbd2c24abd5f2373b7c`,
cost 2,161. It passes 267/267 in each ORT mode, errors 0, truthful shapes,
strict/data-prop, UB=0, and prior fresh seed 171799913 at 5000/5000 in each ORT
mode. A second seed was not run because the strict cost gate had already failed
by +1,333 (score delta -0.9593); it is not a finalist or saved candidate.

### task364

Baseline SHA-256
`2ba1bb84e800b98cdcac9e4d8cb8970d08e532fadf84b64f1b3d88d21ab2a3db`,
cost 685 (memory 683, params 2), is a 468-node cloak graph (196
CenterCropPads, 126 CastLikes). `input_fake` is declared `[1,1,1,1]`, but its
definition `CenterCropPad(input, Shape(input))` guarantees `[1,10,30,30]`.
Known recheck is 266/266 in both modes, errors 0, margin min 1.0; this does not
overrule the false-shape policy.

The ordinary topology/flood control is `scripts/golf/scratch/task364/cand8.onnx`,
SHA-256 `a9876c1abbb330049d793ec1362b2136be51353326784d27224cf21025db0344`,
cost 46,741. It uses Conv/MaxPool/ScatterND/Gather/Where/Equal/Pad, no lookup,
cloak, or giant Einsum; its output is truthfully `[1,10,30,30]`; strict/data-prop
and UB=0 pass. Recheck is 266/266 in both modes, errors 0. Historical independent
evidence is numpy 8000/8000 and ONNX disable-all 2000/2000. The two-seed gate
was not started because the cost is already +46,056 (score delta -4.2230).

### task310

Baseline SHA-256
`f7ad4fb86c5aa78aeb9e3ea8239bf0ba52af1249988a5065992a4e8c58861d79`,
cost 566 (memory 202, params 364), has four TfIdfVectorizers and a final
23-input Einsum. Shapes are truthful and strict/data-prop passes, but the
lookup/giant mechanism is prohibited. It is 266/266 in both modes, errors 0,
margin min 4.0. That known result is not a generalization guarantee: the prior
independent fresh5000 audit is only 4993/5000 under disable-all and 4998/5000
under default. This also matches the documented local/private divergence
lineage (local >20 versus real score 14), so anything below 100% must be rejected.

The exact selector control at
`scripts/golf/loop_7999_13/lane_c9/task310_safe_linear_selector.onnx`, SHA-256
`ad81c541e1164adc0e6a7971ae62a013b79e3bd6a1a3efcbca93b4400084c496`,
is 266/266 and fresh5000/5000 in both ORT modes with errors 0, but costs 633
(score delta -0.11188) and still contains TfIdfVectorizer plus the 23-input
renderer. It is therefore neither cheaper nor admissible, and was not saved.

## Gates and artifacts

Every audited baseline/control passes full checker and strict shape inference;
all Conv-family bias checks report UB=0. The separate truthful-shape audit is
intentionally stronger than inferencer acceptance and rejects the opaque-shape
baselines above. Because no model survived the earlier cost + structural gates,
there was no eligible finalist on which to claim the requested two independent
fresh seeds. Reporting fabricated second-seed evidence would be unsound.

Machine evidence is in `audit.json` and `result.json`; `audit.py` reproduces the
ZIP/true-rule/dual-ORT known audit without writing outside this directory (apart
from transient runtime files under the system temporary area).
