# task012 sub-710 retraining report

## Outcome

`winner = null`. No candidate is eligible for promotion.

The cheapest retained retraining artifact is
`task012_h7w8_known_opt_rejected.onnx`, official cost 570. It is a structurally
clean one-node depthwise Conv, but every runtime configuration scores only
235/265 on the complete known corpus and 176/196 = 89.795918% on the complete
finite default generator geometry. It therefore fails both exact-known and
POLICY90. `submission.zip` and root/stage artifacts were not edited by this
lane.

## Authority and source

The `submission.zip` task012 member is byte-identical to
`artifacts/handcrafted/task012.onnx`:

- SHA-256: `478a310e10fcf0a3e82df943fd6ab43671c47059f8e6eb675bf0004bef576500`
- official cost: 710 = 700 parameters in `[10,1,7,10]` weights plus bias `[10]`
- known: 265/265 in all four ORT configurations
- fresh: 5000/5000 for both seeds in all four ORT configurations
- errors, nonfinite outputs, and output-shape mismatches: 0

The strict lower source
`task012_history_r01_static500_a3640a1525.onnx` has SHA-256
`a3640a15252636a0bc7e3ed3e56353e3d1693c09d3fcadaf4a899c9490531894`
and official cost 500. It reproduces 235/265 known and 176/196 finite default
generator states in every ORT configuration.

## Generator-derived search

The generator has two pre-gravity plus objects, centers at rows 2 and 8,
columns independently in `3..9`, and four gravity orientations. The output
places the center color on the center and diagonal radii 1/2, and the arm
color on axial radii 1/2. This gives 7 x 7 x 4 = 196 finite default latent
states. The cost-500 source is exact for the background and center roles; its
20 failures are arm-role cases with horizontal separation two in gravity 1/3.

This lane added two bounded searches without repeating the historical fixed
alignment campaign:

1. A hard linear feasibility census over all 238 dense 7x8, 7x9, 8x7, and
   9x7 padding layouts on all 265 known examples. Zero layouts make all ten
   channels feasible; the maximum is one feasible channel.
2. A case-level MILP maximizing exact whole-grid cases for the useful 7x8 and
   7x9 alignments. Every reported solve terminated optimal with MIP gap 0.

| Kernel/padding | Dataset | Proven optimum |
|---|---:|---:|
| 7x8, top 3 / left 3 | known | 235/265 |
| 7x8, top 3 / left 3 | default domain | 176/196 |
| 7x9, top 3 / left 3 | known / domain | 235/265 / 176/196 |
| 7x9, top 3 / left 4 | known / domain | 235/265 / 176/196 |
| 7x9, top 3 / left 5 | known / domain | 235/265 / 176/196 |

Thus an added column, the centered offset, and both adjacent asymmetric
offsets do not improve either target. The saved 7x8 and 7x9 models have costs
570 and 640 respectively. Models are explicitly named `_rejected` and are not
normal candidates.

The prior task012 campaign remains the alignment/dilation boundary evidence:

- 2,447,544 nonnegative dilated biased attempts under area 70: none feasible
- 218,962 shifted dense biased attempts under area 70: none feasible
- cost-700 no-bias shifted/dilated/both: 4,312 / 17,054 / 92,366 attempts,
  none feasible
- 702,546 high-area shifted+dilated attempts: none feasible
- direct group=5/2/1 boundary families: patch-label conflicts
- the High47 inventory records the prior complete 1,712-alignment conclusion

These fixed searches were not duplicated.

## Runtime and strict audit

The cost-570 best rejected model was run with:

- ORT_DISABLE_ALL, threads 1
- default optimization, threads 1
- ORT_DISABLE_ALL, threads 4
- default optimization, threads 4

All four give 235/265 known and 176/196 finite-domain exact. The two
independent fresh seeds give 4490/5000 = 89.80% and 4464/5000 = 89.28% in
every configuration. Across known, exhaustive domain, and fresh runs there
are zero errors, nonfinite results, or shape mismatches.

The candidate passes full ONNX checking, strict shape inference with
`data_prop=True`, canonical static `[1,10,30,30]` input/output declarations,
runtime output-shape agreement, standard-domain-only checks, and the Conv
bias-length checker. It contains one Conv, weights `[10,1,7,8]`, bias `[10]`,
no sparse initializer, functions, banned operators, lookup, public-fixture
correction, or shape cloak.

QLinearConv and ConvInteger are not viable sub-710 terminals from the required
float input: quantizing `[1,10,30,30]` materializes a uint8/int8 intermediate
of 9,000 elements before the terminal, so counted intermediate memory alone
already exceeds 710.

## Reproduction and artifacts

Run from the repository root:

```bash
.venv/bin/python scripts/golf/root_task012_sub710_retrain_250/search.py census
.venv/bin/python scripts/golf/root_task012_sub710_retrain_250/search.py milp \
  --dataset known --kh 7 --kw 8 --pt 3 --pl 3 --time-limit 120 \
  --output scripts/golf/root_task012_sub710_retrain_250/milp_known_h7w8_pt3pl3.json
.venv/bin/python scripts/golf/root_task012_sub710_retrain_250/audit.py
```

Authoritative lane artifacts are `search.py`, `audit.py`,
`dense_census.json`, the eight `milp_*.json` solver records, and
`evidence.json`. The ONNX files are rejected diagnostic artifacts only.

Final disposition: `REJECT_KNOWN_AND_POLICY90`; no promotion and verified gain
0.
