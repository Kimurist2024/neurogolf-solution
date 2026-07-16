# shared Concat → single basis / Einsum selector absorption audit

## Outcome

不採用です。`task013/055/099/281` の共有入力を持つ複数 `Concat` を単一basisへ統合し、
必要なselectorを既存Einsum係数へ吸収するexact構成を列挙しましたが、actual competition
profileでstrict-lowerになった候補は0件でした。

- 実測した形状クラス: 72
- 係数anchorの同形別名まで展開した具体候補: 192
- full checker pass: 72/72
- strict shape inference (`strict_mode=True, data_prop=True`) pass: 72/72
- strict-lower: 0/192
- known4 / fresh10000: `NOT_RUN_NO_STRICT_LOWER`
- root `submission.zip` / `all_scores.csv` / authority: 変更なし

## Fixed baselines and best exact costs

| task | fixed baseline | baseline cost | cheapest shared-basis candidate | actual cost | delta |
|---|---|---:|---|---:|---:|
| 013 | `others/71407/task013.onnx`, SHA `97d6a181...` | 356 | C0+C1, direct Qch-axis selectors | 378 | +22 |
| 055 | 8009.46 authority | 234 | H+V, direct PA-axis selectors | 270 | +36 |
| 099 | 8009.46 authority | 398 | retain 3×`one4` + 3×`Lb`, direct all coefficient axes | 582 | +184 |
| 281 | 8009.46 authority | 161 | direct/direct or S2-bridge equivalents | 213 | +52 |

The profiler breakdown for every concrete alias is in `concrete_costs.json`; the full proof metadata,
checker/strict/truthful results, dual-ORT one-known comparisons, SHA, and actual memory/params are in
`screen_results.json`.

## Exhaustive construction families

### task013

The three vectors are `C0=[one,c0]`, `C1=[one,c1]`, and `T=[one,target]`. All three pairwise
merges and the triple merge were built. For each participating vector, its old 2-axis was either:

1. directly embedded into the enlarged `Qch`/`Qor` axis; or
2. retained behind an exact Kronecker bridge `delta[basis,old]` materialized inside that coefficient.

Bridge mode consumes one fresh Einsum label per occurrence. The source equation already uses 49/52
labels, so T-bridge (four occurrences) and simultaneous C0+C1 bridges (four occurrences) are not
representable by ONNX Einsum. All representable combinations were measured. Even the cheapest
C0+C1 direct/direct construction adds 24 coefficient params while saving only two units in Concat
memory, hence 356→378.

### task055

`HZ=[HD,Hs,K]` and `VZ=[VD,Vs,K]` were unified as `[HD,Hs,K,VD,Vs]`. Both PA-axis direct
embedding and PA bridge absorption were enumerated independently for H and V (4 candidates). The
best construction is 234→270.

### task099

`At` contains three `one4` copies and `Ab` contains three `Lb` copies. The shared basis was enumerated
with 1..3 retained copies of each (9 basis layouts, lengths 9..13). A non-injective layout preserves
the old 7-index through a Kronecker bridge absorbed into every possible existing coefficient anchor:

- At anchors: `ST_u`, `FTc`, `RTc`, `DTc`
- Ab anchors: `DB__f0a`, `SB`, `FBc`, `RBc`

For injective 3-copy layouts, direct expansion of all coefficients on the old feature index was also
measured. Anchors with identical tensor shape were profiled once as a shape class and expanded in
`concrete_costs.json`, yielding 169 concrete task099 candidates. The smallest is 398→582.

### task281

`VR=[one,row_inv(2)]` and `VC=[one,col_inv(2)]` were unified as a length-5 basis. Their old
3-index is shared not only by T but also by S2, so the initially tempting T-only direct expansion is
not algebraically valid. The complete exact family therefore uses, independently for R and C:

1. direct expansion of both T and S2 axes;
2. old-index bridge absorbed into T; or
3. old-index bridge absorbed into S2.

All 3×3 combinations were measured; the minimum is 161→213.

## Truthful runtime / finite audit

Task055 and task099 baselines and all candidates are shape-truthful and finite on the traced known
case. Task013 and task281 inherit pre-existing untruthful annotations/nonfinite intermediates from
their fixed baselines:

- task013 baseline and candidates: five declared-vs-actual shape mismatches and 2270 nonfinite traced
  values (`xf`, `x16`, `Rfeat_i8`, `Rfeat`, and output shape are misstated).
- task281 baseline and candidates: two shape mismatches and 1212 nonfinite traced values (`xf`, `x16`).

Those candidates are rejected independently of their already-higher costs. Dual ORT default/disabled
threshold output equality held for all 72 classes on the screening case. Task099 has raw differences
for 28/49 classes because inserting selector axes changes floating reduction order; this does not
affect the algebraic Kronecker proof, but would require deep sign validation if any such candidate
were strict-lower. None is.

## Integrity

- `submission_base_8009.46.zip` SHA-256:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `submission.zip` remained byte-identical to that authority.
- `all_scores.csv` SHA-256 remained
  `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`.
- No Hardmax, lookup, private-zero, approximation, sparse, or external-data mechanism was introduced.

