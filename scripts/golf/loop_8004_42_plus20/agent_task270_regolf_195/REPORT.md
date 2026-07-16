# task270 current8009.46 strict regolf

## Outcome

**採用候補は0件、projected gainは+0.0です。** current authority cost587を
`submission_base_8009.46.zip`から再抽出し、そのgraphだけを基準にtruthful化・packed
detector・renderer rank・initializer/scalar/index/profile fusionを再探索しました。
最安の完全sound/truthful再構成はcost **592**で、authorityより5高いため採用不能です。

root `submission.zip`、`submission_base_8009.46.zip`、`all_scores.csv`、
`others/71407`には書き込んでいません。

| model | SHA-256 | memory + params = cost | known ×4 | shape mismatch | decision |
|---|---|---:|---:|---:|---|
| 8009.46 authority | `0d848124abafda1daf24fe5f779ed5249c9b8b2054854264dde838b05e27a443` | 386 + 201 = **587** | 266/266 | 2 explicit / 4 including dependent CastLike | baseline only |
| truthful direct | `77ecc3c5be720d304482c4c49380c29dc60235b94284d8c5b9c2e0031fbe5cba` | 408 + 200 = **608** | 266/266 | 0 | too expensive |
| truthful packed | `046e15662b85364584c598cb5b00f21ef9bfadbc48a78b72240259962bf1caac` | 404 + 191 = **595** | 266/266 | 0 | too expensive |
| truthful shared-scale | `3d98850eabbb3383d372f31255bfe2a33967d43d56b275c9767e9a1c0cfce4ec` | 402 + 190 = **592** | 266/266 | 0 | sound floor、too expensive |
| saturating `pr2` probe | `9154c38aa6bb6c4937956b305df9b1340090d7cd64700c031cd3e852d6ce621b` | 398 + 190 = **588** | 250/266 | 0 | incorrect |

全候補でfull checker、strict shape inference + data propagation、standard domain、
static shape、finite initializer、function/sparse/nested graphなし、Conv-familyなし
（UB0）を確認しました。knownの全4構成はdisable-all/default × threads1/4です。
全runでruntime error 0、nonfinite 0でした。

## Authority diagnosis

authorityはgenerator ruleを正しく実装し、known 266/266×4、今回のfreshでも
2000/2000×4でした。しかし、`Ridx8` / `Cidx8` の実shape `[2,3]`を
`Shape -> CenterCropPad -> CastLike`で `[1,1]` と宣言しています。

直接traceでは少なくとも次の2 declared/runtime mismatchがあり、依存する`Ridx`/
`Cidx`まで含む既存のfully-typed traceでは4 mismatchです。

- `Ridx8_hid`: declared `[1,1]`, actual `[2,3]`
- `Cidx8_hid`: declared `[1,1]`, actual `[2,3]`

このためcost587はtruthful strict candidateとして再利用できません。さらに最終renderer
は19-input Einsumで、giant-contraction policy markerも残ります。

## Current-derived exact transformations

### 1. Direct truthful indices — cost608

3 nodeの`Shape/CenterCropPad` carrierを除き、`Ridx8/Cidx8 [2,3]`を直接int32へCast。
unusedになった`i32like`も除去しました。raw ruleは保たれshape mismatchは0ですが、
実index tensorが正しく課金されcost608です。

### 2. Packed petal selector — cost595

2-row petal selectorを1-rowのbase-2048 packed selectorへ変換し、count/row/column/
squared-row momentsをuint8 2 laneへ復号しました。これは過去truthful cost595 formと
同じsemantic compilationですが、今回のbuilderはcurrent authorityから再生成して
います。

### 3. Shared center/petal scale — cost592

center codeのbase16をbase2048へ変更し、petal packの既存scaleをcenter row/columnでも
共有しました。centerの`Div+Mod`をlow-byte Cast + high-byte QuantizeLinearへ置換し、
memoryを2、parameterを1削減。packed595から **-3**、truthful cost592になりました。

### 4. Squared-row saturation — cost588、incorrect

`pr2 / 2048 -> Cast(uint8)`をQuantizeLinearへ融合するとfloat intermediate 4 costを
削れますが、QuantizeLinearはuint8範囲外をsaturateし、Castのmodulo-wrapと同値では
ありません。knownは全4構成で **250/266**、同じfirst wrong index 21、error/nonfinite
0でした。authority587をまだ1上回るうえ誤答なので即REJECTです。

## Complete semantic-domain proof for the cost592 sound floor

generator `task_ae3edfdc.py`から規則を再確認しました。

- grid sizeは15、2中心のrow/columnはそれぞれdistinctで2..12。
- 各flowerの4 cardinal petalはpresent/absentの256 mask。
- `sr = Σ(petal_row-center_row)`。
- `qr = Σ(petal_row-center_row)^2`。
- `qr > sr^2` iff 上下petalが両方存在。単独時は`sr`の符号で上下を判定。
- `petal_count-up-down`が水平countで、column momentの符号が左右を判定。

base2048 packはfloat32 integer-exact範囲内です。uint8 lane arithmeticは下記の
全presence maskとfresh coordinate/deltaで検証しました。

### Renderer exhaustive domain

各axisのsemantic stateはpadding `O/O` と、2 flowerの`B/C/P` pairから不可能な
`C/C`を除く8状態、合計9状態です。row×column 81状態のうちgeneratorがcollisionで
rejectする2状態を除いた **79状態**を実rendererで全評価しました。

- generator-reachable in-grid states: 62
- padding込みchecked states: 79
- wrong: **0**
- runtime error / nonfinite: **0 / 0**
- minimum intended positive: **0.72216796875**

### Detector and runtime coverage

- 256/256 presence masks × 4 ORT構成: wrong 0、error 0、nonfinite 0
- fresh seed `27019501`: 1000/1000 × 4 ORT構成
- fresh seed `27019502`: 1000/1000 × 4 ORT構成
- fresh全2000件でcost592とauthorityのraw difference: **0** ×4構成
- known 266/266 ×4構成

従ってcost592はfixture/coordinate lookupではなく、generator全semantic domainに
対応するsound controlです。ただしstrict-lowerではありません。

## Renderer and fusion search

最終rendererはrank6で`A[2,6]` 12 params、`K[1,6,10]` 60 paramsです。rank5なら
12 params削減でき、truthful cost580が狙えます。complete 79-state objectiveで4 run、
合計50,000 stepを探索しましたが、fp16最良でも **15 sign errors**が残り、係数を
候補化できませんでした。rank5不可能性の形式証明ではありませんが、採用可能な
rendererは発見していません。

final Einsumの因数分解は中間`[q,30]` row/column tensorsを課金するため、現行587を
下回りません。Kは5 active colorしか持ちませんが、dense parameter課金であり、
5-channel rendererを10-channel outputへ戻す中間は4500+ costとなります。

Concat/Scatter側では、各axisに3 updateが必要です。truthful formの`[2,3]` uint8
indices、`[2,3]` int32 indices、`[2,30]` fp16 profileはこのformのelement-minimumです。
row/column scatterの統合はbase/update/indexを倍化し、profile memoryも減りません。

initializer監査ではunused initializer 0、byte-identical alias 0、dead node 0、
removable optional output 0でした。`CastLike -> Cast`はshape propagationを露出して
cost626となり、lowerではありません。

## Integrity and evidence

- `submission_base_8009.46.zip` / `submission.zip`:
  `4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927`
- `all_scores.csv`:
  `8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78`
- machine evidence: `audit/result.json`
- reproducible builders/audit: `prepare.py`, `build_variants.py`, `audit_task270.py`
- rank search: `search_renderer_rank.py`, `search/rank5_best.json`
- `others/71407/MANIFEST.json`は他root laneにより複数回更新されたためobserve-only。
  このlaneは一度も書き込んでいません。

Final decision: **no task270 model is merged or staged.**
